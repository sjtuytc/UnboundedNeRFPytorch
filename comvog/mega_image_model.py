import torch.nn as nn
import torch
import pdb
import time
import imageio
import numpy as np
from comvog import utils, dvgo, dcvgo, dmpigo
from comvog import comvog_grid


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


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    return rays_o, rays_d, viewdirs


class MegaImageModel(nn.Module):
    def __init__(self, xyz_min, xyz_max, data_dict, num_voxels_density=0, num_voxels_base_density=0, num_voxels_rgb=0,
                 num_voxels_base_rgb=0, num_voxels_viewdir=0, alpha_init=None, mask_cache_world_size=None, fast_color_thres=0, 
                 bg_len=0.2, contracted_norm='inf', density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={},
                 rgbnet_dim=0, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=4, img_emb_dim=-1, verbose=False,
                 **kwargs):
        super(MegaImageModel, self).__init__()
        self.image = data_dict['images'][0]
        self.density = nn.Parameter(torch.tensor([2.2, 3.1]))
        # imageio.imwrite('test.png', utils.to8b(np.array(self.image)))
        
    def comvog_get_training_rays(self, rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
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
    
    def forward(self, rays_o, rays_d, viewdirs, global_step=None, is_train=False, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        pdb.set_trace()
        
    def gather_training_rays(self, data_dict, images, cfg, i_train, cfg_train, poses, HW, Ks, render_kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        indexs_train = None
        if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega" or cfg.data.dataset_type == "nerfpp":
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_train, imsz = self.comvog_get_training_rays(
            rgb_tr_ori=rgb_tr_ori, train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train], 
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, )
        elif cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=self, render_kwargs=render_kwargs)
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
