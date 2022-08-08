import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo

from . import grid
from .dvgo import Raw2Alpha, Alphas2Weights, render_utils_cuda
from .dmpigo import create_full_step_id


'''Model'''
class DirectBiVoxGO(nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_world_size=None,
                 fast_color_thres=0, bg_preserve=0.5,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0, bg_use_mlp=True,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 **kwargs):
        super(DirectBiVoxGO, self).__init__()
        xyz_min = torch.Tensor(xyz_min)
        xyz_max = torch.Tensor(xyz_max)
        assert len(((xyz_max - xyz_min) * 100000).long().unique()), 'scene bbox must be a cube in DirectBiVoxGO'
        self.register_buffer('scene_center', (xyz_min + xyz_max) * 0.5)
        self.register_buffer('scene_radius', (xyz_max - xyz_min) * 0.5)
        self.register_buffer('xyz_min', torch.Tensor([-1,-1,-1]))
        self.register_buffer('xyz_max', torch.Tensor([1,1,1]))
        self.fast_color_thres = fast_color_thres
        self.bg_preserve = bg_preserve

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = nn.ModuleList([
            grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)
            for _ in range(2)
        ])

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = nn.ModuleList([
                grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
                for _ in range(2)
            ])
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            self.k0_dim = rgbnet_dim
            self.k0 = nn.ModuleList([
                grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
                for _ in range(2)
            ])
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)
            dim0 += self.k0_dim
            self.rgbnet = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
                for _ in range(2)
            ])
            nn.init.constant_(self.rgbnet[0][-1].bias, 0)
            nn.init.constant_(self.rgbnet[1][-1].bias, 0)
            if not bg_use_mlp:
                self.k0[1] = grid.create_grid(
                    k0_type, channels=3, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
                self.rgbnet[1] = None
            print('dvgo: feature voxel grid', self.k0)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = nn.ModuleList([
            grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            for _ in range(2)
        ])

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_world_size': list(self.mask_cache[0].mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density[0].scale_volume_grid(self.world_size)
        self.density[1].scale_volume_grid(self.world_size)
        self.k0[0].scale_volume_grid(self.world_size)
        self.k0[1].scale_volume_grid(self.world_size)

        if np.prod(list(self.world_size)) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            self_alpha = [
                F.max_pool3d(self.activate_density(self.density[0].get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0],
                F.max_pool3d(self.activate_density(self.density[1].get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0],
            ]
            self.mask_cache = nn.ModuleList([
                grid.MaskGrid(
                    path=None, mask=(self_alpha[i]>self.fast_color_thres),
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max)
                for i in range(2)
            ])

        print('dvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache[0].mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache[0].mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache[0].mask.shape[2]),
        ), -1)
        for i in range(2):
            cache_grid_density = self.density[i](cache_grid_xyz)[None,None]
            cache_grid_alpha = self.activate_density(cache_grid_density)
            cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache[i].mask &= (cache_grid_alpha > self.fast_color_thres)

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density[0].total_variation_add_grad(w, w, w, dense_mode)
        self.density[1].total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0[0].total_variation_add_grad(w, w, w, dense_mode)
        self.k0[1].total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = (ori_rays_o - self.scene_center) / self.scene_radius
        rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
        # sample query points in inter scene
        near = 0
        far = 2 * np.sqrt(3)
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        # sample query points in outer scene
        N_outer = int(np.sqrt(3) / stepdist.item() * (1-self.bg_preserve)) + 1
        ray_pts_outer = render_utils_cuda.sample_bg_pts_on_rays(
            rays_o, rays_d, t_max, self.bg_preserve, N_outer)
        return ray_pts, ray_id, step_id, ray_pts_outer

    def _forward(self, ray_pts, viewdirs, interval, N,
                 mask_grid, density_grid, k0_grid, rgbnet=None,
                 ray_id=None, step_id=None, prev_alphainv_last=None):
        # preprocess for bg queries
        if ray_id is None:
            # ray_pts is [N, M, 3] in bg query
            assert len(ray_pts.shape) == 3
            ray_id, step_id = create_full_step_id(ray_pts.shape[:2])
            ray_pts = ray_pts.reshape(-1, 3)

        # skip ray which is already occluded by fg
        if prev_alphainv_last is not None:
            mask = (prev_alphainv_last > self.fast_color_thres)
            ray_id = ray_id.view(N,-1)[mask].view(-1)
            step_id = step_id.view(N,-1)[mask].view(-1)
            ray_pts = ray_pts.view(N,-1,3)[mask].view(-1,3)

        # skip known free space
        mask = mask_grid(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]

        # query for alpha w/ post-activation
        density = density_grid(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for color
        k0 = k0_grid(ray_pts)
        if rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            k0_view = k0
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)

        return dict(
            rgb=rgb, alpha=alpha, weights=weights, alphainv_last=alphainv_last,
            ray_id=ray_id, step_id=step_id)

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, ray_pts_outer = self.sample_ray(
                ori_rays_o=rays_o, ori_rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # query for foreground
        fg = self._forward(
                ray_pts=ray_pts, viewdirs=viewdirs,
                interval=interval, N=N,
                mask_grid=self.mask_cache[0],
                density_grid=self.density[0],
                k0_grid=self.k0[0],
                rgbnet=self.rgbnet[0],
                ray_id=ray_id, step_id=step_id)

        # query for background
        bg = self._forward(
                ray_pts=ray_pts_outer, viewdirs=viewdirs,
                interval=interval, N=N,
                mask_grid=self.mask_cache[1],
                density_grid=self.density[1],
                k0_grid=self.k0[1],
                rgbnet=self.rgbnet[1],
                prev_alphainv_last=fg['alphainv_last'])

        # Ray marching
        rgb_marched_fg = segment_coo(
                src=(fg['weights'].unsqueeze(-1) * fg['rgb']),
                index=fg['ray_id'],
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched_bg = segment_coo(
                src=(bg['weights'].unsqueeze(-1) * bg['rgb']),
                index=bg['ray_id'],
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched = rgb_marched_fg + \
                      fg['alphainv_last'].unsqueeze(-1) * rgb_marched_bg + \
                      (fg['alphainv_last'] * bg['alphainv_last']).unsqueeze(-1) * render_kwargs['bg']
        ret_dict.update({
            'rgb_marched': rgb_marched,
            'alphainv_last': torch.cat([fg['alphainv_last'], bg['alphainv_last']]),
            'weights': torch.cat([fg['weights'], bg['weights']]),
            'raw_alpha': torch.cat([fg['alpha'], bg['alpha']]),
            'raw_rgb': torch.cat([fg['rgb'], bg['rgb']]),
            'ray_id': torch.cat([fg['ray_id'], bg['ray_id']]),
        })

        if render_kwargs.get('render_depth', False):
            # TODO: add bg
            with torch.no_grad():
                depth_fg = segment_coo(
                        src=(fg['weights'] * fg['step_id']),
                        index=fg['ray_id'],
                        out=torch.zeros([N]),
                        reduce='sum')
                depth_bg = segment_coo(
                        src=(bg['weights'] * bg['step_id']),
                        index=bg['ray_id'],
                        out=torch.zeros([N]),
                        reduce='sum')
                depth_fg_last = segment_coo(
                        src=fg['step_id'].float(),
                        index=fg['ray_id'],
                        out=torch.zeros([N]),
                        reduce='max')
                depth_bg_last = segment_coo(
                        src=bg['step_id'].float(),
                        index=bg['ray_id'],
                        out=depth_fg_last.clone(),
                        reduce='max')
                depth = depth_fg + \
                        fg['alphainv_last'] * (1 + depth_fg_last + depth_bg) + \
                        fg['alphainv_last'] * bg['alphainv_last'] * (2 + depth_fg_last + depth_bg_last)
            ret_dict.update({'depth': depth})

        return ret_dict

