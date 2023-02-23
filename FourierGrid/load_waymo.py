'''
Modify from
https://github.com/Kai-46/nerfplusplus/blob/master/data_loader_split.py
'''
import os
import pdb
from tkinter import image_names
import cv2
import glob
import scipy
import imageio
import shutil
import numpy as np
import torch
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation as R
from FourierGrid.common_data_loaders.load_llff import normalize
from FourierGrid.trajectory_generators.waymo_traj import *

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
    

def sample_metadata_by_idxs(metadata, sample_idxs, val_num=5):
    if sample_idxs is None:
        for split in metadata:
            if split != 'train':   # validation is not that important
                sample_idxs = list(range(val_num))
                for one_k in metadata[split]:
                    metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], sample_idxs)
        return metadata
    for split in metadata:
        if split != 'train':   # validation is not that important
            sample_idxs = sample_idxs[:val_num]
        for one_k in metadata[split]:
            metadata[split][one_k] = sample_list_by_idx(metadata[split][one_k], sample_idxs)
    return metadata


def sample_metadata_by_training_ids(metadata, training_ids, assign_pos, assign_rot):
    if training_ids is None:
        return metadata
    for split in metadata:
        if split != 'train':
            continue
        else:
            sample_idxs = []
            for ele in training_ids:
                full_path = f'images_train/{ele}.png'
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


def sort_metadata_by_pos(metadata):
    for split in metadata:
        list_idxs = list(range(len(metadata[split]['position'])))
        # first sort y, then x
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


def find_most_freq_ele(one_list):
    most_freq_ele = max(set(one_list), key = one_list.count)
    freq_count = one_list.count(most_freq_ele)
    return most_freq_ele, freq_count


def save_training_imgs_to_disk(args, cfg, metadata):
    exp_folder = os.path.join(cfg.basedir, cfg.expname)
    data_folder = cfg.data.datadir
    train_imgs = metadata['train']['file_path']
    os.makedirs(exp_folder, exist_ok=True)
    for idx, train_img in enumerate(tqdm(train_imgs)):
        full_data_path = os.path.join(data_folder, train_img)
        assert os.path.exists(full_data_path), f"{full_data_path} does not exist!"
        shutil.copyfile(full_data_path, os.path.join(exp_folder, train_img.split("/")[-1]))
        print(f"img file saved at {exp_folder}.")
    return


def resize_img(train_HW, val_HW, imgs, tr_K, val_K):
    target_h, _ = find_most_freq_ele([hw[0] for hw in train_HW])
    target_w, _ = find_most_freq_ele([hw[1] for hw in train_HW])
    imgs = [cv2.resize(img, dsize=(target_w, target_h), interpolation=cv2.INTER_CUBIC) for img in imgs]
    for idx, one_k in enumerate(tr_K):
        h_before, w_before = train_HW[idx]
        assert h_before == tr_K[idx][1][2] * 2
        assert w_before == tr_K[idx][0][2] * 2
        h_ratio = target_h / h_before
        w_ratio = target_w / w_before
        # alpha x
        tr_K[idx][0][0] = tr_K[idx][0][0] * w_ratio
        # x0
        tr_K[idx][0][2] = tr_K[idx][0][2] * w_ratio
        # alpha y
        tr_K[idx][1][1] = tr_K[idx][1][1] * h_ratio
        # y0
        tr_K[idx][1][2] = tr_K[idx][1][2] * h_ratio
        assert target_w == tr_K[idx][0][2] * 2
        assert target_h == tr_K[idx][1][2] * 2
    for idx, one_k in enumerate(val_K):
        h_before, w_before = val_HW[idx]
        assert h_before == val_K[idx][1][2] * 2
        assert w_before == val_K[idx][0][2] * 2
        h_ratio = target_h / h_before
        w_ratio = target_w / w_before
        # alpha x
        val_K[idx][0][0] = val_K[idx][0][0] * w_ratio
        # x0
        val_K[idx][0][2] = val_K[idx][0][2] * w_ratio
        # alpha y
        val_K[idx][1][1] = val_K[idx][1][1] * h_ratio
        # y0
        val_K[idx][1][2] = val_K[idx][1][2] * h_ratio
        assert target_w == val_K[idx][0][2] * 2
        assert target_h == val_K[idx][1][2] * 2
    train_HW = [[target_h, target_w] for hw in train_HW]
    val_HW = [[target_h, target_w] for hw in val_HW]
    return train_HW, val_HW, imgs, tr_K, val_K


def find_rotations_from_meta(metadata):
    rotations = []
    for idx, c2w in enumerate(metadata['train']['cam2world']):
        rot = np.array(c2w)[:3, :3]
        trans_rot = R.from_matrix(rot).as_euler('yzx', degrees=True)
        rotations.append(trans_rot)
    return rotations
    

def load_waymo(args, cfg, ):
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
    metadata = sample_metadata_by_idxs(metadata, sample_idxs)

    if "training_ids" in cfg.data:
        training_ids = cfg.data.training_ids
        # if 'assign_pos' in cfg.data:
        #     assign_pos, assign_rot = cfg.data.assign_pos, cfg.data.assign_rot
        # else:
        #     assign_pos, assign_rot = None, None
        # if args.program == 'tune_pose':
        #     rand_vec = np.random.rand(3)
        #     lw, up = np.array(cfg.data.search_rot_lower), np.array(cfg.data.search_rot_upper)
        #     cur_rot = lw + (up - lw) * rand_vec
        #     rand_vec = np.random.rand(3)
        #     lw, up = np.array(cfg.data.search_pos_lower), np.array(cfg.data.search_pos_upper)
        #     cur_pos = lw + (up - lw) * rand_vec
        #     assign_pos[cfg.data.tunning_id] = cur_pos.tolist()
        #     assign_rot[cfg.data.tunning_id] = cur_rot.tolist()
        #     args.running_rot = np.round(cur_pos, 4).tolist() + np.round(cur_rot, 2).tolist()
        metadata = sample_metadata_by_training_ids(metadata, training_ids, None, None)
    rotations = find_rotations_from_meta(metadata)
    if args.diffuse:
        for idx, fp in enumerate(metadata['train']['file_path']):
            img_name = fp.split("/")[-1].replace(".png", "")
            diffuse_replace = cfg.diffusion.diff_replace
            if img_name in diffuse_replace:
                img_path = os.path.join(cfg.diffusion.diff_root, diffuse_replace[img_name] + ".png")
                metadata['train']['file_path'][idx] = img_path
    
    # The validation datasets are from the official val split, 
    # but the testing splits are hard-coded sequences (completely novel views)
    tr_cam_idx, val_cam_idx = metadata['train']['cam_idx'], metadata['val']['cam_idx']
    cam_idxs = tr_cam_idx + val_cam_idx
    train_pos, val_pos = metadata['train']['position'], metadata['val']['position']
    positions = train_pos + val_pos
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

    if args.save_train_imgs:
        save_training_imgs_to_disk(args, cfg, metadata)
    train_HW, val_HW, imgs, tr_K, val_K = resize_img(train_HW, val_HW, imgs, tr_K, val_K)

    # Create the test split
    te_c2w, test_HW, test_K, test_cam_idxs, test_pos = \
        gen_rotational_trajs(args, cfg, metadata, tr_c2w, train_HW, tr_K, tr_cam_idx, train_pos, 
                       rotate_angle=data_cfg.test_rotate_angle)
    # te_c2w, test_HW, test_K, test_cam_idxs = \
    #     gen_straight_trajs(metadata, tr_c2w, train_HW, tr_K, tr_cam_idx, train_pos, 
    #                    rotate_angle=data_cfg.test_rotate_angle)
    # dummy test paths
    # te_c2w, test_K, test_HW, test_cam_idxs = val_c2w, val_K, val_HW, val_cam_idx
    # TODO: consider removing the so-called test split.
    for _ in te_c2w:
        i_split[2].append(loop_id)
        loop_id += 1
    for c2w in te_c2w:
        poses.append(np.array(c2w).reshape(4,4))
    
    # Bundle all the data
    all_K = np.array(tr_K + val_K + test_K)
    HW = np.array(train_HW + val_HW + test_HW)
    poses = np.stack(poses, 0)
    if load_img:
        imgs = np.stack(imgs)
        
    # note test_cam_idxs can be inaccurate because it may be varied!
    cam_idxs += test_cam_idxs
    render_poses = te_c2w
    return imgs, poses, render_poses, HW, all_K, cam_idxs, i_split


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far


def load_waymo_data(args, cfg):
    data_cfg = cfg.data
    K, depths = None, None
    near_clip = None
    images, poses, render_poses, HW, K, cam_idxs, i_split = load_waymo(args, cfg)
    print(f"Loaded waymo dataset.")
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
        poses=poses, render_poses=render_poses, images=images, depths=depths, cam_idxs=cam_idxs, irregular_shape=irregular_shape
    )
    data_dict['poses'] = torch.tensor(data_dict['poses']).float()
    data_dict['images'] = torch.tensor(data_dict['images']).float()
    return data_dict
