'''
Modify from
https://github.com/Kai-46/nerfplusplus/blob/master/data_loader_split.py
'''
import os
import pdb
import glob
from turtle import onkey
import scipy
import imageio
import numpy as np
import torch
from tqdm import tqdm
import json

from yono.common_data_loaders.load_llff import normalize

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


def sample_list_by_idx(one_list, idxs):
    return [one_list[idx] for idx in idxs]
    
    
def sample_metadata_by_cam(metadata, cam_idx):
    for split in metadata:
        sample_idxs = []
        for idx, cam_id in enumerate(metadata[split]['cam_idx']):
            if cam_id == cam_idx:
                sample_idxs.append(idx)
        for one_k in metadata[split]:
            metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], sample_idxs)
    return metadata
    

def sample_metadata_by_idxs(metadata, sample_idxs):
    if sample_idxs is None:
        return metadata
    for split in metadata:
        if split != "train":
            # TODO: remove this for the validation dataset
            sample_idxs = [1, 2, 3, 4, 5]
        for one_k in metadata[split]:
            metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], sample_idxs)
    return metadata


def sort_metadata_by_pos(metadata):
    for split in metadata:
        list_idxs = list(range(len(metadata[split]['position'])))
        sorted_idxs = sorted(zip(list_idxs, metadata[split]['position']), key=lambda row: (row[1][1], row[1][0]))
        sorted_idxs = [i for i, j in sorted_idxs]
        for one_k in metadata[split]:
            metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], sorted_idxs)
    return metadata


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def load_waymo(args, data_cfg, rerotate=True, normalize_pose=True, recenter_pose=True):
    basedir = data_cfg.datadir
    with open(os.path.join(basedir, f'metadata.json'), 'r') as fp:
        metadata = json.load(fp)
    if 'sample_cam' in data_cfg:
        metadata = sample_metadata_by_cam(metadata, data_cfg['sample_cam'])
    if args.sample_num > 0:
        sample_idxs = list(range(args.sample_num))
    elif 'sample_idxs' in data_cfg:
        sample_idxs = data_cfg['sample_idxs']
    else:
        sample_idxs = None
    metadata = sort_metadata_by_pos(metadata)
    metadata = sample_metadata_by_idxs(metadata, sample_idxs)
    tr_cam_idx, val_cam_idx = metadata['train']['cam_idx'], metadata['test']['cam_idx']
    cam_idxs = tr_cam_idx + val_cam_idx
    positions = metadata['train']['position'] + metadata['test']['position']
    tr_im_path = metadata['train']['file_path']
    te_im_path = metadata['test']['file_path']
    # te_im_path = waymo_load_img_list(os.path.join(basedir, 'images_test'))
    tr_c2w, te_c2w = metadata['train']['cam2world'], metadata['test']['cam2world']
    tr_K, te_K = metadata['train']['K'], metadata['test']['K']
    
    all_K = np.array(tr_K + te_K)
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
    if args.program == "gen_trace":
        imgs = tr_im_path + te_im_path  # do not load all the images
    else:
        imgs = []
        for path in tqdm(tr_im_path):
            imgs.append(imageio.imread(os.path.join(basedir, path)) / 255.)
        for path in tqdm(te_im_path):
            imgs.append(imageio.imread(os.path.join(basedir, path)) / 255.) 
        imgs = np.stack(imgs)
    # Bundle all data
    # imgs = np.stack(imgs, 0)
    
    # Pose filters
    poses = np.stack(poses, 0)
    if normalize_pose:
        poses[:, :3, 3] = (poses[:, :3, 3] - poses[:, :3, 3].min()) /  (poses[:, :3, 3].max() - poses[:, :3, 3].min())
    # if recenter_pose:
        # poses = recenter_poses(poses)
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
    render_poses = None
    train_HW = np.array([[metadata['train']['height'][i], metadata['train']['width'][i]] for i in range(len(metadata['train']['height']))]).tolist()
    test_HW = np.array([[metadata['test']['height'][i], metadata['test']['width'][i]] for i in range(len(metadata['test']['height']))]).tolist()
    HW = np.array(train_HW + test_HW)
    return imgs, poses, render_poses, HW, all_K, cam_idxs, i_split, positions


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
    images, poses, render_poses, HW, K, cam_idxs, i_split, positions = load_waymo(args, data_cfg)
    print(f"Loaded waymo dataset.")
    i_train, i_val, i_test = i_split
    near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0.02)
    
    # load near and far parameters
    if "near_clip" in data_cfg:
        near_clip = data_cfg['near_clip']
    if 'near' in data_cfg:
        near = data_cfg['near']
    if 'far' in data_cfg:
        far = data_cfg['far']
    Ks = np.array(K)
    # # Cast intrinsics to right types
    # H, W, focal = hwf
    # H, W = int(H), int(W)
    # hwf = [H, W, focal]
    # HW = np.array([im.shape[:2] for im in images])
    irregular_shape = False

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
    # hwf = None
    data_dict = dict(
        HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths, cam_idxs=cam_idxs, 
        positions=positions, irregular_shape=irregular_shape
    )
    data_dict['poses'] = torch.tensor(data_dict['poses']).float()
    data_dict['images'] = torch.tensor(data_dict['images']).float()
    return data_dict
