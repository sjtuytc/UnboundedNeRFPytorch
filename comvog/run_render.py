import os
import pdb
import imageio
import torch
from tqdm import tqdm, trange
import numpy as np
from comvog import utils, dvgo, dcvgo, dmpigo
from comvog.comvog_model import ComVoGModel
from comvog.utils import resize_and_to_8b
from comvog.arf import ARF
import matplotlib.pyplot as plt


@torch.no_grad()
def render_viewpoints(cfg, model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    tqdm_bar = tqdm(render_poses)
    for i, c2w in enumerate(tqdm_bar):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega" or cfg.data.dataset_type == "nerfpp":
            indexs = torch.zeros_like(rays_o)
            indexs.copy_(torch.tensor(i).long().to(rays_o.device))  # add image index
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **{**render_kwargs, "indexs": ind}).items() if k in keys}
                for ro, rd, vd, ind in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), 
                                           viewdirs.split(8192, 0), indexs.split(8192, 0))
            ]
        else:
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
            ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        tqdm_bar.set_description(f"Rendering video with frame shape: {rgb.shape}.")
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def run_render(args, cfg, data_dict, device, debug=True, add_info=""):
    # initilize stylizer when needed
    if 'arf' in cfg:
        stylizer = ARF(cfg, data_dict, device)
    else:
        stylizer = None
    model_class = ComVoGModel                 # only support ComVoGModel currently
    merged_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last_merged.tar')
    use_merged = os.path.exists(merged_ckpt_path)
    if use_merged:
        merged_model = utils.load_model(model_class, merged_ckpt_path).to(device)
    else:
        merged_model = None
    render_viewpoints_kwargs = {
        'model': None,
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': cfg.fine_model_and_render.stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
            'render_depth': True,
        }
    }
    
    # block-by-block rendering
    if args.block_num > 1 and not use_merged:
        print("Merging trained blocks ...")
        ckpt_paths = [os.path.join(cfg.basedir, cfg.expname, f'fine_last_{i}.tar') for i in range(args.block_num)]
        if args.render_train:
            model_class = ComVoGModel                 # only support ComVoGModel currently
            ckpt_paths = [os.path.join(cfg.basedir, cfg.expname, f'fine_last_{i}.tar') for i in range(args.block_num)]
            train_save_dir = os.path.join(cfg.basedir, cfg.expname, f'render_train_fine_last')
            os.makedirs(train_save_dir, exist_ok=True)
            print('All results are dumped into', train_save_dir)
            all_rgbs = []
            all_training_indexs = data_dict['i_train'].copy()
            for idx, cp in enumerate(ckpt_paths):
                args.running_block_id = idx
                s, e = idx * args.num_per_block, (idx + 1) * args.num_per_block
                # Here we assume the i_train's order follows the block order.
                data_dict['i_train'] = all_training_indexs[s:e]
                model = utils.load_model(model_class, cp).to(device)
                render_viewpoints_kwargs['model'] = model
                rgbs, depths, bgmaps = render_viewpoints(cfg=cfg, render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']], Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=train_save_dir, dump_images=args.dump_images, eval_ssim=args.eval_ssim, 
                eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
                if stylizer is not None:
                    rgbs, _ = stylizer.match_colors_for_image_set(rgbs, train_save_dir)
                all_rgbs += rgbs.tolist()
            save_all_rgbs = np.array(all_rgbs)
            save_name = 'video.rgb.mp4'
            if stylizer is not None:
                save_name = f'video.rgb.style.{cfg.arf.style_id}.mp4'
            imageio.mimwrite(os.path.join(train_save_dir, save_name), utils.to8b(save_all_rgbs), fps=15, quality=8)

        if args.render_test:
            model_class = ComVoGModel                 # only support ComVoGModel currently
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_fine_last')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            if data_dict['i_test'][0] >= len(data_dict['images']):  # gt images are not provided
                gt_imgs = None
            else:
                gt_imgs = [data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']]
            ckpt_paths = [os.path.join(cfg.basedir, cfg.expname, f'fine_last_{i}.tar') for i in range(args.block_num)]
            all_rgbs = []
            all_test_indexs = data_dict['i_test'].copy()
            for idx, cp in enumerate(ckpt_paths):
                args.running_block_id = idx
                s, e = idx * args.num_per_block, (idx + 1) * args.num_per_block
                # Here we assume the i_test's order follows the block order.
                data_dict['i_test'] = all_test_indexs[s:e]
                model = utils.load_model(model_class, cp).to(device)
                render_viewpoints_kwargs['model'] = model
                rgbs, depths, bgmaps = render_viewpoints(
                cfg=cfg, render_poses=data_dict['poses'][data_dict['i_test']], HW=data_dict['HW'][data_dict['i_test']], 
                Ks=data_dict['Ks'][data_dict['i_test']], gt_imgs=gt_imgs,
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
                all_rgbs += rgbs.tolist()
            save_all_rgbs = np.array(all_rgbs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(save_all_rgbs), fps=15, quality=8)
        return
    # rendering merged model or normal cases
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega" or cfg.data.dataset_type == "nerfpp":
            model_class = ComVoGModel
        elif cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        if use_merged:
            model = merged_model
        else:
            model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }
        geometry_path = os.path.join(cfg.basedir, cfg.expname, f'geometry.npz')
        if model_class == ComVoGModel:
            model.export_geometry_for_visualize(geometry_path)

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        
        rgbs, depths, bgmaps = render_viewpoints(cfg=cfg,
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), resize_and_to_8b(rgbs, res=(800, 608)), fps=30, quality=8)
        # TODO: make the depth visualization work with resize
        # imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), resize_and_to_8b(1 - depths / np.max(depths), res=(800, 608)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        if data_dict['i_test'][0] >= len(data_dict['images']):  # gt images are not provided
            gt_imgs = None
        else:
            gt_imgs = [data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']]
        rgbs, depths, bgmaps = render_viewpoints(
                cfg=cfg, render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=gt_imgs,
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                cfg=cfg,
                render_poses=data_dict['render_poses'],
                # use the hw and ks from the test splits
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        if 'running_rot' in args:
            vid_name = add_info[:5] + "_" + str(args.running_rot) + '.rgb.mp4'
        else:
            vid_name = 'video.rgb.mp4'
        print(f"Rendering video saved at: {os.path.join(testsavedir, vid_name)}.")
        imageio.mimwrite(os.path.join(testsavedir, vid_name), utils.to8b(rgbs), fps=30, quality=8)
        depths_vis = depths * (1-bgmaps) + bgmaps
        mask = bgmaps < 0.1
        if not mask.max():
            print("depth img cannot be rendered because of the threshold!")
        else:
            dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
            depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
            imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
    return
