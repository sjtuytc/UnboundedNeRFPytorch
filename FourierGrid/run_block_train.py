import time
import torch
import torch.nn.functional as F
import os,pdb
import copy
import numpy as np
from tqdm import tqdm, trange
from FourierGrid.bbox_compute import compute_bbox_by_cam_frustrm, compute_bbox_by_coarse_geo
from FourierGrid import utils, dvgo, dcvgo, dmpigo
from FourierGrid.FourierGrid_model import FourierGridModel
from FourierGrid.load_everything import load_existing_model
from torch_efficient_distloss import flatten_eff_distloss
from FourierGrid.run_export_bbox import run_export_bbox_cams
from FourierGrid.run_export_coarse import run_export_coarse
from FourierGrid.FourierGrid_model import FourierGrid_get_training_rays


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_new_model(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, device):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
    verbose = args.block_num <= 1
    FourierGrid_datasets = ["waymo", "mega", "nerfpp", "tankstemple"]
    if cfg.data.dataset_type in FourierGrid_datasets or cfg.model == 'FourierGrid':
        if verbose:
            print(f'Waymo scene_rep_reconstruction ({stage}): \033[96m Use FourierGrid model. \033[0m')
        model_kwargs['sample_num'] = args.sample_num
        model = FourierGridModel(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels, verbose=verbose,
            **model_kwargs)
    elif cfg.data.ndc:
        if verbose:
            print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        if verbose:
            print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        if verbose:
            print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    verbose = args.block_num <= 1
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0, verbose=verbose)
    return model, optimizer


# init batch rays sampler
def gather_training_rays(data_dict, images, cfg, i_train, cfg_train, poses, HW, Ks, model, render_kwargs):
    if data_dict['irregular_shape']:
        rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
    else:
        rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

    indexs_train = None
    FourierGrid_datasets = ["waymo", "mega", "nerfpp", "tankstemple"]
    if cfg.data.dataset_type in FourierGrid_datasets or cfg.model == 'FourierGrid':
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_train, imsz = FourierGrid_get_training_rays(
        rgb_tr_ori=rgb_tr_ori, train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train], 
        ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
        flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    elif cfg_train.ray_sampler == 'in_maskcache':
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, render_kwargs=render_kwargs)
    elif cfg_train.ray_sampler == 'flatten':
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
            rgb_tr_ori=rgb_tr_ori,
            train_poses=poses[i_train],
            HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    else:
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
            rgb_tr=rgb_tr_ori,
            train_poses=poses[i_train],
            HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
    batch_index_sampler = lambda: next(index_generator)
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_train, imsz, batch_index_sampler


def block_scene_rep_reconstruction(block_id, train_index, args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None, load_model=None):
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    
    # render_poses are removed because they are unused
    HW, Ks, near, far, poses, images = [data_dict[k] for k in ['HW', 'Ks', 'near', 'far', 'poses', 'images']]

    # reload if ckpt exists
    if block_id >= 0:
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last_{block_id}.tar')
    else:  # merged
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last_merged.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:  # not supported
        raise RuntimeError("ft_path is not supported.")
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None
    FourierGrid_datasets = ["waymo", "mega", "nerfpp", "tankstemple"]
    # init model and optimizer
    if reload_ckpt_path is None:  # create new model
        if load_model is not None:
            model = load_model
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0, verbose=False)
        else:
            print(f'scene_rep_reconstruction ({stage}): train from scratch')
            model, optimizer = create_new_model(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, device)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[train_index,:3,3], near)
    elif cfg.data.dataset_type in FourierGrid_datasets or cfg.model == 'FourierGrid':
        print(f'scene_rep_reconstruction ({stage}): reload FourierGrid model from {reload_ckpt_path}')
        model, optimizer, start = args.ckpt_manager.load_existing_model(args, cfg, cfg_train, reload_ckpt_path, device=device)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existing_model(args, cfg, cfg_train, reload_ckpt_path, device=device)
    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_tr, imsz, batch_index_sampler = \
        gather_training_rays(data_dict, images, cfg, train_index, cfg_train, poses, HW, Ks, model, render_kwargs)
    
    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            elif isinstance(model, FourierGridModel):
                model.scale_volume_grid(cur_voxels)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            # torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            if indexs_tr is not None:
                indexs = indexs_tr[sel_i]
            else:
                indexs = None
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            if indexs_tr is not None:
                indexs = indexs_tr[sel_b, sel_r, sel_c]
            else:
                indexs = None
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            if indexs is not None:
                indexs = indexs.to(device)

        render_kwargs['indexs'] = indexs  # to avoid change the model interface
        # volumetric rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, is_train=True,
            **render_kwargs)
        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []
        
        # saving in the middle is forbidden in FourierGrid training.

    # final save
    if global_step != -1:
        FourierGrid_datasets = ["waymo", "mega", "nerfpp", "tankstemple"]
        if cfg.data.dataset_type in FourierGrid_datasets or cfg.model == 'FourierGrid':
            args.ckpt_manager.save_model(global_step, model, optimizer, last_ckpt_path)
        else:               
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, last_ckpt_path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)
    
    model.cpu()
    del model
            
            
def run_training_pipeline(args, cfg, data_dict, block_id, all_training_indexs, cur_block_indexs, export_cam=True, export_geometry=True, load_model=None):
    if block_id >= 0:
        print(f"Training block id: {block_id}.")
    else:
        print('Training: start.')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))
    data_dict['i_train'] = all_training_indexs
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    print('FourierGrid train: skip coarse geometry searching')
    coarse_ckpt_path = None

    # export cameras and geometries for debugging
    if export_cam:
        run_export_bbox_cams(args=args, cfg=cfg, data_dict=data_dict, save_path=os.path.join(cfg.basedir, cfg.expname, 'cam.npz'))
    if export_geometry and cfg.coarse_train.N_iters > 0:
        run_export_coarse(args=args, cfg=cfg, device=device, save_path=os.path.join(cfg.basedir, cfg.expname, 'cam_coarse.npz'))

    # fine detail reconstruction
    eps_fine = time.time()
    FourierGrid_datasets = ["waymo", "mega", "nerfpp", "tankstemple"]
    if cfg.coarse_train.N_iters == 0 or cfg.data.dataset_type in FourierGrid_datasets or cfg.model == 'FourierGrid':
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres, device=device, args=args, cfg=cfg)
    block_scene_rep_reconstruction(
            block_id=block_id, train_index=cur_block_indexs, args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path,
            load_model=load_model)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    if block_id >= 0:
        print('train: reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    if block_id >= 0:
        print('train: finish (eps time', eps_time_str, ')')


def run_block_train_and_merge(args, cfg, data_dict, export_cam=False, export_geometry=False):
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last_merged.tar')
    all_training_indexs = data_dict['i_train'].copy()
    if not os.path.exists(last_ckpt_path):
        for block_id in range(args.block_num):
            s, e = block_id * args.num_per_block, (block_id + 1) * args.num_per_block
            cur_block_indexs = all_training_indexs[s:e]
            run_training_pipeline(args, cfg, data_dict, block_id, all_training_indexs, cur_block_indexs, export_cam=export_cam, export_geometry=export_geometry)
        torch.cuda.empty_cache()                    # empty cache because it will not be used in the future.
        merged_model = args.ckpt_manager.merge_blocks(args, cfg, device)
    else:
        merged_model = None
    run_training_pipeline(args, cfg, data_dict, -1, all_training_indexs, all_training_indexs, export_cam=export_cam, 
                          export_geometry=export_geometry, load_model=merged_model)
    print("Training finished. Run multi-block rendering.")
