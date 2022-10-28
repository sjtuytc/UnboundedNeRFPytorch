import time
import torch
import torch.nn.functional as F
import os,pdb
import copy
import numpy as np
from tqdm import tqdm, trange
from comvog.bbox_compute import compute_bbox_by_cam_frustrm, compute_bbox_by_coarse_geo
from comvog import utils, dvgo, dcvgo, dmpigo
from comvog.comvog_model import ComVoGModel
from comvog.load_everything import load_existing_model
from torch_efficient_distloss import flatten_eff_distloss
from comvog.run_export_bbox import run_export_bbox_cams
from comvog.run_export_coarse import run_export_coarse
from comvog.comvog_model import comvog_get_training_rays, FourierMSELoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def create_new_model(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, device):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels_density = model_kwargs.pop('num_voxels_density')
    num_voxels_rgb = model_kwargs.pop('num_voxels_rgb')
    if len(cfg_train.pg_scale):
        num_voxels_density = int(num_voxels_density / (2**len(cfg_train.pg_scale)))
        num_voxels_rgb = int(num_voxels_rgb / (2**len(cfg_train.pg_scale)))
    verbose = False

    if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega" or cfg.data.dataset_type == "nerfpp":
        if verbose:
            print(f'Waymo scene_rep_reconstruction ({stage}): \033[96m Use ComVoG model. \033[0m')
        model_kwargs['sample_num'] = args.sample_num
        model = ComVoGModel(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels_density=num_voxels_density, num_voxels_rgb=num_voxels_rgb, verbose=verbose,
            **model_kwargs)
    elif cfg.data.ndc or cfg.data.unbounded_inward:
        raise RuntimeError("These settings are no longer supported in ComVoG.")
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
    if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega" or cfg.data.dataset_type == "nerfpp":
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_train, imsz = comvog_get_training_rays(
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


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # init
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    
    # render_poses are removed because they are unused
    HW, Ks, near, far, i_train, i_val, i_test, poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, device)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    elif cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega" or cfg.data.dataset_type == "nerfpp":
        print(f'scene_rep_reconstruction ({stage}): reload ComVoG model from {reload_ckpt_path}')
        model, optimizer, start = args.ckpt_manager.load_existing_model(args, cfg, cfg_train, reload_ckpt_path, device=device)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existing_model(args, cfg, cfg_train, reload_ckpt_path, device=device)
    
    # init loss
    fourier_mse_loss = FourierMSELoss(num_freqs=7, logscale=True)
    
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
        gather_training_rays(data_dict, images, cfg, i_train, cfg_train, poses, HW, Ks, model, render_kwargs)
    
    # view-count-based learning rate
    if cfg_train.pervoxel_lr and reload_ckpt_path is None:
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

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):
        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels_density = int(cfg_model.num_voxels_density / (2**n_rest_scales))
            cur_voxels_rgb = int(cfg_model.num_voxels_rgb / (2**n_rest_scales))
            if isinstance(model, ComVoGModel):
                model.scale_volume_grid(cur_voxels_density, cur_voxels_rgb)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

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
        elif cfg_train.ray_sampler == 'random':  # fixed function
            if len(rgb_tr.shape) == 3:
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
                assert len(rgb_tr.shape) == 2, "tgb_tr's shape is not correct."
                sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
                sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
                target = rgb_tr[sel_b]
                rays_o = rays_o_tr[sel_b]
                rays_d = rays_d_tr[sel_b]
                viewdirs = viewdirs_tr[sel_b]
                if indexs_tr is not None:
                    indexs = indexs_tr[sel_b]
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
        # forward model here and get rendered results
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, is_train=True,
            **render_kwargs)
        optimizer.zero_grad(set_to_none=True)
        psnr_loss = F.mse_loss(render_result['rgb_marched'], target)
        freq_loss = fourier_mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(psnr_loss.detach())
        loss = cfg_train.weight_main * psnr_loss + cfg_train.weight_freq * freq_loss
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

        # # update lr, continuously decaying
        # decay_steps = cfg_train.lrate_decay * 1000
        # decay_factor = 0.1 ** (1/decay_steps)
        # for i_opt_g, param_group in enumerate(optimizer.param_groups):
        #     param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'training iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []
        
        if global_step==1+start:  # test saving function at start
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega" or cfg.data.dataset_type == "nerfpp":
                args.ckpt_manager.save_model(global_step, model, optimizer, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'model_kwargs': model.get_kwargs(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    # final save
    if global_step != -1:
        if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega": 
            args.ckpt_manager.save_model(global_step, model, optimizer, last_ckpt_path)
        else:               
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, last_ckpt_path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def run_train(args, cfg, data_dict, export_cam=True, export_geometry=True):
    # init
    running_block_id = args.running_block_id
    if running_block_id >= 0:
        print(f"Training block id: {running_block_id}.")
    else:
        print('Training: start.')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching (originally only for inward bounded scenes, extended to support waymo)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # export cameras and geometries for debugging
    if export_cam:
        run_export_bbox_cams(args=args, cfg=cfg, data_dict=data_dict, save_path=os.path.join(cfg.basedir, cfg.expname, 'cam.npz'))
    if export_geometry and cfg.coarse_train.N_iters > 0:
        run_export_coarse(args=args, cfg=cfg, device=device, save_path=os.path.join(cfg.basedir, cfg.expname, 'cam_coarse.npz'))

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0 or cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega":
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres, device=device, args=args, cfg=cfg)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    if running_block_id >= 0:
        print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    if running_block_id >= 0:
        print('train: finish (eps time', eps_time_str, ')')
