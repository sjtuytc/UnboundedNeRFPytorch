import glob
import os
import json
import numpy as np
import torch
from kornia import create_meshgrid
from typing import Tuple
from numpy.linalg import tensorsolve
import pdb
import numpy.linalg as la
import scipy.linalg as spla
from tqdm import tqdm


def get_cam_rays(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)  # (H, W, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions.numpy()


def get_Rotate(img_rays_cam2world, pose_path="c2w_Poses.json"):
    Rotates = {}
    cur_meta_file = pose_path
    for idx, img_name in enumerate(img_rays_cam2world):
        print(f"Solving the {idx + 1}/{len(img_rays_cam2world)} image's pose...")
        rays = img_rays_cam2world[img_name]
        cam_ray_dir = np.array(rays['cam_ray_dir'].reshape(-1, 3))
        world_ray_dir = np.array(rays['world_ray_dir'].reshape(-1, 3))
        
        world_r123 = np.mat(world_ray_dir[:, :1]).reshape(-1, 1)
        world_r456 = np.mat(world_ray_dir[:, 1:2]).reshape(-1, 1)
        world_r789 = np.mat(world_ray_dir[:, 2:3]).reshape(-1, 1)

        cam_dir = np.mat(cam_ray_dir)
        r123 = np.linalg.lstsq(cam_dir, world_r123, rcond=None)[0]
        r456 = np.linalg.lstsq(cam_dir, world_r456, rcond=None)[0]
        r789 = np.linalg.lstsq(cam_dir, world_r789, rcond=None)[0]

        R = np.zeros([3, 3])
        R[0:1, :] = r123.T
        R[1:2, :] = r456.T
        R[2:3, :] = r789.T

        '''
        T = np.zeros([3, 4])
        T[:, :3] = R
        T[:, 3:] = t.reshape(3, 1)
        '''
        Rotates[img_name] = R.tolist()

        loss = world_ray_dir - cam_ray_dir @ R 

        print(f"loss:\t{np.absolute(loss).mean()}")

        with open(cur_meta_file, "w") as fp:
            json.dump(Rotates, fp)
            fp.close()

    return Rotates

def get_rotate_one_image(cam_ray_dir, world_ray_dir):
    b_matrix = cam_ray_dir = cam_ray_dir.reshape(-1, 3)
    A_matrix = world_ray_dir = world_ray_dir.reshape(-1, 3)

    world_r123 = np.mat(world_ray_dir[:, :1]).reshape(-1, 1)
    world_r456 = np.mat(world_ray_dir[:, 1:2]).reshape(-1, 1)
    world_r789 = np.mat(world_ray_dir[:, 2:3]).reshape(-1, 1)

    cam_dir = np.mat(cam_ray_dir)
    r123 = np.linalg.lstsq(cam_dir, world_r123, rcond=None)[0]
    r456 = np.linalg.lstsq(cam_dir, world_r456, rcond=None)[0]
    r789 = np.linalg.lstsq(cam_dir, world_r789, rcond=None)[0]

    R = np.zeros([3, 3])
    R[0:1, :] = r123.T
    R[1:2, :] = r456.T
    R[2:3, :] = r789.T

    '''
    T = np.zeros([3, 4])
    T[:, :3] = R
    T[:, 3:] = t.reshape(3, 1)
    '''
    R_loss = world_ray_dir - cam_ray_dir @ R.T
    print(f"Pose loss:\t{np.absolute(R_loss).mean()}")
    return R.tolist()

if __name__ == "__main__":
    root_dir = "data/pytorch_block_nerf_dataset"
    pose_save_dir = "data/c2w_poses.json"

    with open(os.path.join(root_dir, 'json/train.json'), 'r') as fp:
        meta = json.load(fp)

    img_rays_cam2world = {}
    poses = {}
    for idx, img_name in tqdm(enumerate(meta)):
        print(f"Handling the {idx}/{len(meta)} image...")
        img_info = meta[img_name]
        img_w = img_info['width']
        img_h = img_info['height']
        # 首先构建K
        K = {}
        K = np.zeros((3, 3), dtype=np.float32)
        # fx=focal,fy=focal,cx=img_w/2,cy=img_h/2
        K[0, 0] = img_info['intrinsics'][0]
        K[1, 1] = img_info['intrinsics'][1]
        K[0, 2] = img_w * 0.5
        K[1, 2] = img_h * 0.5
        K[2, 2] = 1

        cam_ray_dir = get_cam_rays(img_h, img_w, K) # 归一化后的相机坐标系光线方向向量
        world_ray_dir = np.array(np.load(os.path.join(root_dir, "images", f"{img_name}_ray_dirs.npy"), mmap_mode='r'))
        pose = get_rotate_one_image(cam_ray_dir, world_ray_dir)
        poses[img_name] = pose
        with open(pose_save_dir, "w") as fp:
            json.dump(poses, fp)
            fp.close()

    print("Finding poses finished!")
