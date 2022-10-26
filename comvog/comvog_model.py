import os
import time
import functools
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo
from comvog.grid import DenseGrid

from . import comvog_grid
from .dvgo import Raw2Alpha, Alphas2Weights
from .dmpigo import create_full_step_id
import ub360_utils_cuda


def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def comvog_get_training_rays(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    indexs_tr = torch.zeros_like(rgb_tr)  # image indexs
    imsz = []
    top = 0
    cur_idx = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        indexs_tr[top:top+n].copy_(torch.tensor(cur_idx).long().to(DEVICE))
        cur_idx += 1
        imsz.append(n)
        top += n
    assert top == N
    eps_time = time.time() - eps_time
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_tr, imsz


class NeRFPosEmbedding(nn.Module):
    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(NeRFPosEmbedding, self).__init__()

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]

        return torch.cat(out, -1)


'''Model'''
class ComVoGModel(nn.Module):
    def __init__(self, xyz_min, xyz_max, num_voxels=0, num_voxels_base=0, alpha_init=None,
                 mask_cache_world_size=None, fast_color_thres=0, bg_len=0.2, contracted_norm='inf',
                 density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={},
                 rgbnet_dim=0, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=4, img_emb_dim=-1, verbose=False,
                 **kwargs):
        super(ComVoGModel, self).__init__()
        xyz_min = torch.Tensor(xyz_min)
        xyz_max = torch.Tensor(xyz_max)
        assert len(((xyz_max - xyz_min) * 100000).long().unique()), 'scene bbox must be a cube in DirectContractedVoxGO'
        self.register_buffer('scene_center', (xyz_min + xyz_max) * 0.5)
        self.register_buffer('scene_radius', (xyz_max - xyz_min) * 0.5)
        # xyz_min/max are the boundary that separates fg and bg scene in NDC.
        self.register_buffer('xyz_min', torch.Tensor([-1,-1,-1]) - bg_len)
        self.register_buffer('xyz_max', torch.Tensor([1,1,1]) + bg_len)
        if isinstance(fast_color_thres, dict):
            self._fast_color_thres = fast_color_thres
            self.fast_color_thres = fast_color_thres[0]
        else:
            self._fast_color_thres = None
            self.fast_color_thres = fast_color_thres
        self.bg_len = bg_len
        self.contracted_norm = contracted_norm
        self.verbose = verbose

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        if self.verbose:
            print('ComVoG: set density bias shift to', self.act_shift)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = comvog_grid.create_grid(
            density_type, channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=True,
            config=self.density_config)
        
        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 
            'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        
        self.img_embed_dim = img_emb_dim
        self.sample_num = kwargs['sample_num']

        if img_emb_dim > 0 and self.sample_num > 0:    # use apperance embeddings
            self.img_embeddings = nn.Embedding(num_embeddings=self.sample_num, 
                                        embedding_dim=self.img_embed_dim)
        else:
            self.img_embeddings = None
            self.img_embed_dim = 0

        # rgbnet configurations
        self.vector_grid = False
        if self.vector_grid:
            self.k0_dim = 9  # rgb * 3
            self.k0 = comvog_grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=False,
                config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)
            dim0 += 3  # real k0 dim is 3
            dim0 += self.img_embed_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('ComVoG: feature voxel grid', self.k0)
            print('ComVoG: mlp', self.rgbnet)
        elif rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = comvog_grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,  use_nerf_pos=False,
                config=self.k0_config)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            self.k0_dim = rgbnet_dim
            self.k0 = comvog_grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=True,
                config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)  # view freq dim
            dim0 += self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            if self.verbose:
                print('ComVoG: feature voxel grid', self.k0)
                print('ComVoG: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = comvog_grid.MaskGrid(
            path=None, mask=mask,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.world_len = self.world_size[0].item()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        if self.verbose:
            print('ComVoG: voxel_size      ', self.voxel_size)
            print('ComVoG: world_size      ', self.world_size)
            print('ComVoG: voxel_size_base ', self.voxel_size_base)
            print('ComVoG: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'contracted_norm': self.contracted_norm,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'sample_num': self.sample_num, 
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)
        nearest_dist = torch.stack([(self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)for co in cam_o.split(10)]).amin(0)
        self.density.grid[nearest_dist[None,None] <= near_clip] = -100
        
    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('ComVoG: voxel_count_views start')
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu())+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = comvog_grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += (ones.grid.grad > 1)
        eps_time = time.time() - eps_time
        print('ComVoG: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count
    
    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        if self.verbose:
            print('ComVoG: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        if self.verbose:
            print('ComVoG: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = comvog_grid.MaskGrid(
                path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('ComVoG: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        if self.verbose:
            print(f'ComVoG: update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        # TODO: Check or remove this function. This is untested and unused for now.
        if self.verbose:
            print('ComVoG: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = comvog_grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, indexs, inner_mask, t, rays_d_e = self.sample_ray(
                        ori_rays_o=rays_o.to(device), ori_rays_d=rays_d.to(device),
                        **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        if self.verbose:
            print(f'ComVoG: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        eps_time = time.time() - eps_time
        if self.verbose:
            print(f'ComVoG: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays: central sampling.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        # NDC coordinates
        rays_o = (ori_rays_o - self.scene_center) / self.scene_radius  
        rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
        N_inner = int(2 / (2+2*self.bg_len) * self.world_len / stepsize) + 1
        N_outer = N_inner
        t_boundary = 1.5  # default t_boundary=2
        b_inner = torch.linspace(0, t_boundary, N_inner+1)
        b_outer = t_boundary / torch.linspace(1, 1/128, N_outer+1)
        t = torch.cat([
            (b_inner[1:] + b_inner[:-1]) * 0.5,
            (b_outer[1:] + b_outer[:-1]) * 0.5,
        ])
        ray_pts = rays_o[:,None,:] + rays_d[:,None,:] * t[None,:,None]
        if self.contracted_norm == 'inf':
            norm = ray_pts.abs().amax(dim=-1, keepdim=True)
        elif self.contracted_norm == 'l2':
            norm = ray_pts.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        seperate_boundary = 1.0
        B = 1 + self.bg_len
        order = 1  # default order = 1
        A = B * (seperate_boundary**order) - seperate_boundary ** (order + 1)
        ray_pts = torch.where(
            norm<=seperate_boundary,
            ray_pts,
            ray_pts / norm * (B - A/ (norm ** order))
        )
        assert 'indexs' in render_kwargs, "The image indexes should be provided in render kwargs in the ComVoG model."
        if 'indexs' in render_kwargs:
            indexs = render_kwargs['indexs'].unsqueeze(1).repeat(1, ray_pts.shape[1], 1)
        if self.vector_grid:
            rays_d_extend = rays_d[:,None,:] * torch.ones_like(t[None,:,None])
        else: 
            rays_d_extend = None
        inner_mask = norm<=seperate_boundary  # this variable is not important
        return ray_pts, indexs, inner_mask.squeeze(-1), t, rays_d_extend

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, is_train=False, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only support point queries in [N, 3] format'
        if isinstance(self._fast_color_thres, dict) and global_step in self._fast_color_thres:
            if self.verbose:
                print(f'ComVoG: update fast_color_thres {self.fast_color_thres} => {self._fast_color_thres[global_step]}')
            self.fast_color_thres = self._fast_color_thres[global_step]

        ret_dict = {}
        num_rays = len(rays_o)
        # sample points on rays
        ray_pts, ray_indexs, inner_mask, t, rays_d_e = self.sample_ray(
                ori_rays_o=rays_o, ori_rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        n_max = len(t)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])

        # skip oversampled points outside scene bbox
        mask = inner_mask.clone() # default

        # changing shapes, only needed when the above procedures are commented out
        t = t[None].repeat(num_rays, 1)
        
        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        
        # apply fast color thresh
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            if rays_d_e is not None:
                rays_d_e = rays_d_e[mask]
            ray_indexs = ray_indexs[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            # changed because the above masking functions are removed
            ray_id = ray_id[mask.flatten()]
            step_id = step_id[mask.flatten()]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, num_rays)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            # print(f"Masked ratio: {1 - mask.sum() / mask.numel()}.")
            ray_pts = ray_pts[mask]        
            if rays_d_e is not None:
                rays_d_e = rays_d_e[mask]    
            ray_indexs = ray_indexs[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            weights = weights[mask]
        else:
            ray_pts = ray_pts.reshape(-1, ray_pts.shape[-1])
            weights = weights.reshape(-1)
            inner_mask = inner_mask.reshape(-1)

        # query for color
        if self.vector_grid:
            k0 = self.k0.vector_forward(ray_pts, rays_d_e)
        else:
            k0 = self.k0(ray_pts)
        
        if self.vector_grid:
            # ComVoG inference procedure
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)
        elif self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)

        # Ray marching, rendering equations here.
        rgb_marched = segment_coo(
                src=weights.unsqueeze(-1) * rgb,
                index=ray_id,
                out=torch.zeros([num_rays, 3]),
                reduce='sum')
        
        if render_kwargs.get('rand_bkgd', False):
            rgb_marched += (alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched))
        
        s = 1 - 1/(1+t)  # [0, inf] => [0, 1]
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_density': density,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'step_id': step_id,
            'n_max': n_max,
            't': t,
            's': s,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * s),
                        index=ray_id,
                        out=torch.zeros([num_rays]),
                        reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict
    
    def export_geometry_for_visualize(self, save_path):
        with torch.no_grad():
            dense_grid = self.density.get_dense_grid()
            alpha = self.activate_density(dense_grid).squeeze().cpu().numpy()
            color_grid = self.k0.get_dense_grid()
            rgb = torch.sigmoid(color_grid).squeeze().permute(1,2,3,0).cpu().numpy()
            np.savez_compressed(save_path, alpha=alpha, rgb=rgb)
            print(f"Geometry is saved at {save_path}.")


class DistortionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, s, n_max, ray_id):
        n_rays = ray_id.max()+1
        interval = 1/n_max
        w_prefix, w_total, ws_prefix, ws_total = ub360_utils_cuda.segment_cumsum(w, s, ray_id)
        loss_uni = (1/3) * interval * w.pow(2)
        loss_bi = 2 * w * (s * w_prefix - ws_prefix)
        ctx.save_for_backward(w, s, w_prefix, w_total, ws_prefix, ws_total, ray_id)
        ctx.interval = interval
        return (loss_bi.sum() + loss_uni.sum()) / n_rays

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        w, s, w_prefix, w_total, ws_prefix, ws_total, ray_id = ctx.saved_tensors
        interval = ctx.interval
        grad_uni = (1/3) * interval * 2 * w
        w_suffix = w_total[ray_id] - (w_prefix + w)
        ws_suffix = ws_total[ray_id] - (ws_prefix + w*s)
        grad_bi = 2 * (s * (w_prefix - w_suffix) + (ws_suffix - ws_prefix))
        grad = grad_back * (grad_bi + grad_uni)
        return grad, None, None, None

distortion_loss = DistortionLoss.apply
