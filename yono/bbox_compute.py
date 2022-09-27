import torch
import numpy as np
from yono import utils, dvgo
import time
from yono.load_everything import load_existing_model
from tqdm import tqdm
import pdb


def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in tqdm(zip(HW[i_train], Ks[i_train], poses[i_train]), total=len(HW[i_train])):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


def _compute_bbox_by_cam_frustrm_waymo(cfg, HW, Ks, poses, i_train, near_clip):
    xs, ys, zs = [], [], []
    for (H, W), K, c2w in tqdm(zip(HW[i_train], Ks[i_train], poses[i_train]), total=len(HW[i_train])):
        xs.append(c2w[:, 3][0].item())
        ys.append(c2w[:, 3][1].item())
        zs.append(c2w[:, 3][2].item())
    zmin, zmax = min(zs), max(zs)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # xyz_min = [xmin - 0.20, ymin-0.05, zmin-0.05]
    # xyz_max = [xmax + 0.03, ymax+0.05, zmax+0.05]
    # ~19
    # xyz_min = [xmin - 0.05, ymin-0.01, zmin-0.01]
    # xyz_max = [xmax + 0.05, ymax+0.01, zmax+0.01]
    xyz_min = [xmin - 0.001, ymin - 0.001, zmin - 0.001]
    xyz_max = [xmax + 0.001, ymax + 0.001, zmax + 0.001]
    xyz_min, xyz_max = torch.tensor(xyz_min), torch.tensor(xyz_max)
    # # Find a tightest cube that cover all camera centers
    # near = -0.1
    # far = 0.1
    # xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    # xyz_max = -xyz_min
    
    # for (H, W), K, c2w in tqdm(zip(HW[i_train], Ks[i_train], poses[i_train]), total=len(HW[i_train])):
    #     rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
    #             H=H, W=W, K=K, c2w=c2w,
    #             ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
    #             flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    #     near_points, far_points = rays_o + rays_d * near, rays_o + rays_d * far
    #     # pts = rays_o + rays_d * near_clip
    #     # xyz_min = torch.minimum(xyz_min, near_points.amin((0,1)))
    #     # xyz_max = torch.maximum(xyz_max, far_points.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.dataset_type == "waymo":
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_waymo(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))
    elif cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres, device, args, cfg):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    # model = utils.load_model(model_class, model_path)
    model, _, _ = load_existing_model(args, cfg, cfg.fine_train, model_path, device=device)
    model.to(device)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max
