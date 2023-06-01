import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import render_utils_cuda
import total_variation_cuda


def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return FourierGrid(**kwargs)
    else:
        raise NotImplementedError


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


''' 
Dense 3D grid
'''
class FourierGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, use_nerf_pos, fourier_freq_num, config):
        super(FourierGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        if use_nerf_pos:
            self.nerf_pos_num_freq = fourier_freq_num
            self.nerf_pos = NeRFPosEmbedding(num_freqs=self.nerf_pos_num_freq)
            self.pos_embed_output_dim = 1 + self.nerf_pos_num_freq * 2
            grid_channels = channels * self.pos_embed_output_dim
        else:
            self.nerf_pos_num_freq = -1
            self.pos_embed_output_dim = -1
            self.nerf_pos = None
            grid_channels = channels
        self.grid = nn.Parameter(torch.zeros([1, grid_channels, *world_size]))
    
    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if self.nerf_pos is not None:
            pos_embed = self.nerf_pos(ind_norm).squeeze()
            out = 0
            for i in range(self.pos_embed_output_dim):
                cur_grid = self.grid.squeeze()[i * self.channels:(i+1) * self.channels]
                cur_pos_embed = pos_embed[:, 3*i:3*(i+1)].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                out += F.grid_sample(cur_grid.unsqueeze(0), cur_pos_embed, mode='bilinear', align_corners=True)
            out /= self.pos_embed_output_dim
            out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)  # only works for channels = 1
        else:
            out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
            out = out.reshape(self.channels,-1).T.reshape(*shape, self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


def compute_tensorf_feat(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, f_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = F.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = torch.cat([
        xy_feat * z_feat,
        xz_feat * y_feat,
        yz_feat * x_feat,
    ], dim=-1)
    feat = torch.mm(feat, f_vec)
    return feat


def compute_tensorf_val(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = F.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = (xy_feat * z_feat).sum(-1) + (xz_feat * y_feat).sum(-1) + (yz_feat * x_feat).sum(-1)
    return feat


''' Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'

