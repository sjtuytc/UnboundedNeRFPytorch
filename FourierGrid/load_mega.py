'''
Modify from
https://github.com/Kai-46/nerfplusplus/blob/master/data_loader_split.py
'''
import os
import pdb
import glob
from tkinter.tix import HList
import scipy
import imageio
import numpy as np
import torch
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation as R
from FourierGrid.common_data_loaders.load_llff import normalize
from FourierGrid.trajectory_generators.waymo_traj import *
from FourierGrid.trajectory_generators.mega_traj import *

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
    # img files
    img_files = find_files('{}'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        img_files = img_files[::skip]
    else:
        raise RuntimeError(f"Cannot find image files at {split_dir}.")

    return img_files


def sample_list_by_idx(one_list, idxs):
    # allow idxs to be out of range
    return [one_list[idx] for idx in idxs if idx < len(one_list)]
    
    
def sample_metadata_by_cam(metadata, cam_idx):
    for split in metadata:
        sample_idxs = []
        for idx, cam_id in enumerate(metadata[split]['cam_idx']):
            if cam_id == cam_idx:
                sample_idxs.append(idx)
        for one_k in metadata[split]:
            metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], sample_idxs)
    return metadata
    

def find_most_freq_ele(one_list):
    most_freq_ele = max(set(one_list), key = one_list.count)
    freq_count = one_list.count(most_freq_ele)
    return most_freq_ele, freq_count


def sample_metadata_by_shape(metadata):
    # only leave images with the same shape
    w_list, h_list = metadata['train']['width'], metadata['train']['height']
    wh_list = list(zip(w_list, h_list))
    wh_most_freq, _ = find_most_freq_ele(wh_list)
    for split in metadata:
        cur_wh_list = list(zip(metadata[split]['width'], metadata[split]['height']))
        filtered_idx = [idx for idx in range(len(cur_wh_list)) if cur_wh_list[idx] == wh_most_freq]
        for one_k in metadata[split]:
            metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], filtered_idx)
    return metadata
    

def sample_metadata_by_idxs(metadata, sample_idxs):
    if sample_idxs is None:
        return metadata
    for split in metadata:
        for one_k in metadata[split]:
            metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], sample_idxs)
    return metadata


def sort_metadata_by_pos(metadata):
    # find the central image.
    train_positions = []
    for c2w in metadata['train']['cam2world']:
        pos = np.array(c2w)[:3, 3]
        train_positions.append(pos)
    center_pos = np.mean(train_positions, 0)
    dis = [np.linalg.norm(pos-center_pos) for pos in train_positions]
    center_id = dis.index(np.min(dis))
    # sort images by position
    list_idxs = list(range(len(dis)))
    sorted_idxs = sorted(zip(list_idxs, dis), key=lambda row: (row[1]))
    sorted_idxs = [idx[0] for idx in sorted_idxs]
    for one_k in metadata['train']:
        metadata['train'][one_k] = sample_list_by_idx(metadata['train'][one_k], sorted_idxs)
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


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w
    

def sample_metadata_by_training_ids(metadata, training_ids, assign_pos, assign_rot):
    if training_ids is None:
        return metadata
    for split in metadata:
        if split != 'train':
            continue
        else:
            sample_idxs = []
            for ele in training_ids:
                full_path = f'images_train/{ele}.jpg'
                if full_path in metadata['train']['file_path']:
                    sample_idxs.append(metadata['train']['file_path'].index(full_path))
            assert len(sample_idxs) > 0, "No image is selected by training id!"
            for one_k in metadata[split]:
                metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], sample_idxs)
            if assign_pos is not None:
                for ele in assign_pos:
                    full_path = f'images_train/{ele}.png'
                    index = metadata[split]['file_path'].index(full_path)
                    metadata[split]['position'][index] = assign_pos[ele]
                    temp_c2w = np.array(metadata[split]['cam2world'][index])
                    # update position
                    temp_c2w[:3, -1] = np.array(metadata[split]['position'][index])
                    trans_rot = R.from_matrix(temp_c2w[:3, :3]).as_euler('yzx', degrees=True)
                    print(full_path, trans_rot)
                    new_rot = assign_rot[ele]
                    r = R.from_euler('yzx', new_rot, degrees=True)
                    temp_c2w[:3, :3] = r.as_matrix()
                    metadata[split]['cam2world'][index] = temp_c2w.tolist()
    return metadata


def load_mega(args, cfg, ):
    data_cfg = cfg.data
    load_img = False if args.program == "gen_trace" else True
    basedir = data_cfg.datadir
    with open(os.path.join(basedir, f'metadata.json'), 'r') as fp:
        metadata = json.load(fp)
    if 'sample_cam' in data_cfg:
        metadata = sample_metadata_by_cam(metadata, data_cfg['sample_cam'])
    if args.sample_num > 0:
        sample_idxs = list(range(0, args.sample_num * data_cfg['sample_interval'], data_cfg['sample_interval']))
        assert args.sample_num * data_cfg['sample_interval'] < len(metadata['train']['file_path']), \
            f"Not enough data to train with given sample interval: {data_cfg['sample_interval']}!"
    elif 'sample_idxs' in data_cfg:
        sample_idxs = data_cfg['sample_idxs']
    else:
        sample_idxs = None
    metadata = sort_metadata_by_pos(metadata)
    metadata = sample_metadata_by_shape(metadata)  # sample the most common shape
    metadata = sample_metadata_by_idxs(metadata, sample_idxs)
    if "training_ids" in cfg.data:
        training_ids = cfg.data.training_ids
        metadata = sample_metadata_by_training_ids(metadata, training_ids, None, None)
    # The validation datasets are from the official val split, 
    # but the testing splits are hard-coded sequences (completely novel views)
    tr_im_path, val_im_path = metadata['train']['file_path'], metadata['val']['file_path']
    tr_c2w, val_c2w = metadata['train']['cam2world'], metadata['val']['cam2world']
    tr_K, val_K = metadata['train']['K'], metadata['val']['K']

    # Determine split id list
    i_split = [[], [], []]
    loop_id = 0
    for _ in tr_c2w:
        i_split[0].append(loop_id)
        loop_id += 1
    for _ in val_c2w:
        i_split[1].append(loop_id)
        loop_id += 1

    # Load camera poses
    poses = []
    for c2w in tr_c2w:
        poses.append(np.array(c2w).reshape(4,4))
    for c2w in val_c2w:
        poses.append(np.array(c2w).reshape(4,4))

    # Load images
    if not load_img:
        imgs = tr_im_path + val_im_path  # do not load all the images
    else:
        imgs = []
        print(f"Loading all the images to disk.")
        for path in tqdm(tr_im_path):
            imgs.append(imageio.imread(os.path.join(basedir, path)) / 255.)
        for path in tqdm(val_im_path):
            imgs.append(imageio.imread(os.path.join(basedir, path)) / 255.) 

    train_HW = np.array([[metadata['train']['height'][i], metadata['train']['width'][i]] 
                         for i in range(len(metadata['train']['height']))]).tolist()
    val_HW = np.array([[metadata['val']['height'][i], metadata['val']['width'][i]] 
                       for i in range(len(metadata['val']['height']))]).tolist()

    te_c2w, test_HW, test_K = \
        gen_rotational_trajs(args, cfg, metadata, tr_c2w, train_HW, tr_K, 
                       rotate_angle=data_cfg.test_rotate_angle)
    # dummy test paths
    # te_c2w, test_HW, test_K = gen_dummy_trajs(metadata, tr_c2w, train_HW, tr_K)
    
    for _ in te_c2w:
        i_split[2].append(loop_id)
        loop_id += 1
    for c2w in te_c2w:
        poses.append(np.array(c2w).reshape(4,4))
    
    # Bundle all the data
    all_K = np.array(tr_K + val_K + test_K)
    hws = train_HW + val_HW + test_HW
    hws = [[int(hw[0]), int(hw[1])] for hw in hws]
    HW = np.array(hws)
    poses = np.stack(poses, 0)
    if load_img:
        imgs = np.stack(imgs)
    render_poses = te_c2w
    return imgs, poses, render_poses, HW, all_K, i_split


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far


def load_mega_data(args, cfg):
    data_cfg = cfg.data
    K, depths = None, None
    near_clip = None
    images, poses, render_poses, HW, K, i_split = load_mega(args, cfg)
    print(f"Loaded MEGA dataset.")
    i_train, i_val, i_test = i_split
    near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0.02)  # not used too much in fact
    
    # load near and far parameters
    if "near_clip" in data_cfg:
        near_clip = data_cfg['near_clip']
    if 'near' in data_cfg:
        near = data_cfg['near']
    if 'far' in data_cfg:
        far = data_cfg['far']
    Ks = np.array(K)
    irregular_shape = False
    data_dict = dict(
        HW=HW, Ks=Ks, near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses, images=images, depths=depths, irregular_shape=irregular_shape
    )
    data_dict['poses'] = torch.tensor(data_dict['poses']).float()
    # TODO: change to device = cuda to avoid load-on-the-fly costs
    data_dict['images'] = torch.tensor(data_dict['images'], device='cpu').float()
    return data_dict
