'''
Modify from
https://github.com/Kai-46/nerfplusplus/blob/master/data_loader_split.py
'''
import os
import pdb
import glob
from posixpath import splitdrive
import scipy
import imageio
import numpy as np
import torch
from tqdm import tqdm
import json

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
def find_files(dir, exts):
    if os.path.isdir(dir):
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []


def waymo_load_img_list(split_dir, skip=1):
    # # camera parameters files
    # intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    # pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])

    # intrinsics_files = intrinsics_files[::skip]
    # pose_files = pose_files[::skip]
    # cam_cnt = len(pose_files)

    # img files
    img_files = find_files('{}'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        img_files = img_files[::skip]
    else:
        raise RuntimeError(f"Cannot find image files at {split_dir}.")

    return img_files


def rerotate_poses(poses, render_poses):
    poses = np.copy(poses)
    centroid = poses[:,:3,3].mean(0)

    poses[:,:3,3] = poses[:,:3,3] - centroid

    # Find the minimum pca vector with minimum eigen value
    x = poses[:,:3,3]
    mu = x.mean(0)
    cov = np.cov((x-mu).T)
    ev , eig = np.linalg.eig(cov)
    cams_up = eig[:,np.argmin(ev)]
    if cams_up[1] < 0:
        cams_up = -cams_up

    # Find rotation matrix that align cams_up with [0,1,0]
    R = scipy.spatial.transform.Rotation.align_vectors(
            [[0,-1,0]], cams_up[None])[0].as_matrix()

    # Apply rotation and add back the centroid position
    poses[:,:3,:3] = R @ poses[:,:3,:3]
    poses[:,:3,[3]] = R @ poses[:,:3,[3]]
    poses[:,:3,3] = poses[:,:3,3] + centroid
    render_poses = np.copy(render_poses)
    render_poses[:,:3,3] = render_poses[:,:3,3] - centroid
    render_poses[:,:3,:3] = R @ render_poses[:,:3,:3]
    render_poses[:,:3,[3]] = R @ render_poses[:,:3,[3]]
    render_poses[:,:3,3] = render_poses[:,:3,3] + centroid
    return poses, render_poses


def load_waymo(args, basedir, rerotate=True):
    with open(os.path.join(basedir, f'metadata.json'), 'r') as fp:
        metadata = json.load(fp)
    tr_im_path = waymo_load_img_list(os.path.join(basedir, 'images_train'))
    assert len(tr_im_path) > 0, f"Images are not found in {basedir}"
    te_im_path = waymo_load_img_list(os.path.join(basedir, 'images_test'))
    # assert len(tr_K) == len(tr_c2w) and len(tr_K) == len(tr_im_path)
    # assert len(te_K) == len(te_c2w) and len(te_K) == len(te_im_path)
    tr_c2w, te_c2w = metadata['train']['cam2world'], metadata['test']['cam2world']
    tr_K, te_K = metadata['train']['K'], metadata['test']['K']
    all_K = tr_K + te_K
    # Determine split id list
    i_split = [[], []]
    i = 0
    for _ in tr_c2w:
        i_split[0].append(i)
        i += 1
    for _ in te_c2w:
        i_split[1].append(i)
        i += 1

    # # Load camera intrinsics. Assume all images share a intrinsic.
    # K_flatten = np.loadtxt(tr_K[0])
    # for path in tr_K:
    #     assert np.allclose(np.loadtxt(path), K_flatten)
    # for path in te_K:
    #     assert np.allclose(np.loadtxt(path), K_flatten)
    # K = K_flatten.reshape(4,4)[:3,:3]

    # Load camera poses
    poses = []
    for c2w in tr_c2w:
        poses.append(np.array(c2w).reshape(4,4))
    for c2w in te_c2w:
        poses.append(np.array(c2w).reshape(4,4))

    # Load images
    imgs = tr_im_path + te_im_path
    # imgs = []
    # for path in tqdm(tr_im_path):
    #     imgs.append(imageio.imread(path) / 255.)
    # for path in tqdm(te_im_path):
    #     imgs.append(imageio.imread(path) / 255.) 
        
    # Bundle all data
    # imgs = np.stack(imgs, 0)
    poses = np.stack(poses, 0)
    # Add the val split as the test split
    i_split.append(i_split[1])
    # H, W = imgs.shape[1:3]
    # focal = K[[0,1], [0,1]].mean()

    # # Generate movie trajectory
    # render_poses_path = sorted(glob.glob(os.path.join(basedir, 'camera_path', 'pose', '*txt')))
    # render_poses = []
    # for path in render_poses_path:
    #     render_poses.append(np.loadtxt(path).reshape(4,4))
    # render_poses = np.array(render_poses)
    # render_K = np.loadtxt(glob.glob(os.path.join(basedir, 'camera_path', 'intrinsics', '*txt'))[0]).reshape(4,4)[:3,:3]
    # render_poses[:,:,0] *= K[0,0] / render_K[0,0]
    # render_poses[:,:,1] *= K[1,1] / render_K[1,1]
    # TODO: test this function
    # if rerotate:
    #     poses, render_poses = rerotate_poses(poses, render_poses)

    # render_poses = torch.Tensor(render_poses)
    H, W = metadata['train']['height'][0], metadata['train']['width'][0]
    HW = np.array([[H, W] for i in imgs])
    if args.sample_num > 0:
        i_split[0] = i_split[0][:args.sample_num]
    return imgs, poses, None, HW, all_K, i_split
    # return imgs, poses, render_poses, [H, W, focal], K, i_split


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far


def load_waymo_data(args, data_cfg):
    K, depths = None, None
    near_clip = None
    images, poses, render_poses, HW, K, i_split = load_waymo(args, data_cfg.datadir)
    print(f"Loaded waymo dataset.")
    i_train, i_val, i_test = i_split
    near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0.02)
    near = 0
    Ks = np.array(K)
    
    # # Cast intrinsics to right types
    # H, W, focal = hwf
    # H, W = int(H), int(W)
    # hwf = [H, W, focal]
    # HW = np.array([im.shape[:2] for im in images])
    # irregular_shape = (images.dtype is np.dtype('object'))

    # if K is None:
    #     K = np.array([
    #         [focal, 0, 0.5*W],
    #         [0, focal, 0.5*H],
    #         [0, 0, 1]
    #     ])

    # if len(K.shape) == 2:
    #     Ks = K[None].repeat(len(poses), axis=0)
    # else:
    #     Ks = K

    # render_poses = render_poses[..., :4]

    data_dict = dict(
        hwf=[], HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
    )
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict
