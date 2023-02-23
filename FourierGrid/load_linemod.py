import pdb
import os
import pickle
import imageio
import numpy as np
import torch
import json
import cv2
import math
import random
from FourierGrid import utils
import open3d as o3d
from tqdm import tqdm
from pytorch3d.io import load_obj, load_ply
from pytorch3d.io import IO
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from FourierGrid.pose_utils.visualization import *
from FourierGrid.pose_utils.projection import *
from FourierGrid.pose_utils.model_operations import *
from FourierGrid.pose_utils.image_operators import *
from FourierGrid.pose_utils.pose_operators import *


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far


def get_most_bbox_size(seq_info, data_root):
    split = 'train'
    skip = 1
    imgs, poses, ks = [], [], []
    width_max = 0
    height_max = 0
    for idx, one_info in enumerate(seq_info[::skip]):
        one_k = one_info['K']
        ks.append(one_k)
        fname = os.path.join(data_root, 'LM6d_converted/LM6d_refine', one_info['rgb_observed_path'])
        one_img = imageio.imread(fname)
        label_fname = fname.replace('color', 'label')
        label_img = imageio.imread(label_fname)
        xmin, xmax, ymin, ymax = get_bbox_from_mask(label_img)
        if xmax - xmin > width_max:
            width_max = xmax - xmin
        if ymax - ymin > height_max:
            height_max = ymax - ymin
    return width_max, height_max


def gen_rotational_trajs(args, poses, test_num=100):
    init_c2w = poses[0]
    rotate_interval = 1
    all_poses = [init_c2w]
    for i in range(test_num - 1):
        cur_c2w = all_poses[-1].copy()
        rotate_r = R.from_euler('z', rotate_interval, degrees=True)
        cur_c2w[:3] = np.matmul(rotate_r.as_matrix(), cur_c2w[:3])
        all_poses.append(cur_c2w)
    all_poses = np.stack(all_poses, axis=0)
    return all_poses


def se3_q2m(se3_q):
    assert se3_q.size == 7
    se3_mx = np.zeros((3, 4))
    # quat = se3_q[0:4] / LA.norm(se3_q[0:4])
    quat = se3_q[:4]
    R = quat2mat(quat)
    se3_mx[:, :3] = R
    se3_mx[:, 3] = se3_q[4:]
    return se3_mx


def t_norm(pose):
    return np.linalg.norm(pose[:3, -1])
    

def uniform_three(kps):
    return_kps = np.float32([kps[0], kps[len(kps)//2], kps[-1]])
    return return_kps
    

def sort_pose_by_rot(seq_info):
    pick_pose = seq_info[0]['gt_pose']
    idx_diff = []
    for idx, one_info in enumerate(seq_info):
        cur_pose = one_info['gt_pose']
        ang_diff = cal_pose_rot_diff(pick_pose, cur_pose)
        idx_diff.append([idx, ang_diff])
    sorted(idx_diff, key=lambda row:row[1])
    return idx_diff


def split_seq_info_global(seq_info, train_ratio=0.95, val_num=20):
    # train_ratio can be set to near 1.0 because we can use all synthetic data
    total_num = len(seq_info)
    train_num = int(total_num * train_ratio)
    random.seed(1)
    all_indexs = [i for i in range(total_num)]
    train_indexs = random.sample(all_indexs, train_num)
    test_indexs = [ind for ind in all_indexs if ind not in train_indexs]
    train_info, val_info, test_info = [], [], []
    for idx, one_info in enumerate(seq_info):
        if idx in test_indexs:
            test_info.append(one_info)
        else:
            train_info.append(one_info)
    val_info = train_info[:val_num]
    if len(test_indexs) < 1:
        test_info = val_info
    return train_info, val_info, test_info
    


def split_seq_info_syn_canonical(syn_gt, train_ratio=0.95, val_num=20):
    # train_ratio can be set to near 1.0 because we can use all synthetic data
    list_syn_gt = [syn_gt[str(key)] for key in range(len(syn_gt))]
    total_num = len(list_syn_gt)
    train_num = int(total_num * train_ratio)
    random.seed(0)
    all_indexs = [i for i in range(total_num)]
    train_indexs = random.sample(all_indexs, train_num)
    test_indexs = [ind for ind in all_indexs if ind not in train_indexs]
    train_info, val_info, test_info = [], [], []
    for idx, one_info in enumerate(list_syn_gt):
        one_info = one_info[0]
        one_info['image_id'] = idx
        if idx in test_indexs:
            test_info.append(one_info)
        else:
            train_info.append(one_info)
    val_info = train_info[:val_num]
    if len(test_indexs) < 1:
        test_info = val_info
    return train_info, val_info, test_info


def load_synthetic_data_canonical(args, cfg, sample_num=None):
    data_root = cfg.data.datadir
    seq_name = cfg.data.seq_name
    # load model
    ply_path = os.path.join(data_root, 'models', seq_name, seq_name + ".xyz")
    obj_m = o3d.io.read_point_cloud(ply_path, format='xyz')
    obj_m = np.asarray(obj_m.points)
    syn_gt_p = os.path.join(data_root, 'train', str(cfg.data.seq_id).zfill(6), 'scene_gt.json')
    syn_gt = json.load(open(syn_gt_p))
    syn_cam_p = os.path.join(data_root, 'train', str(cfg.data.seq_id).zfill(6), 'scene_camera.json')
    syn_cam = json.load(open(syn_cam_p))
    train_info, val_info, test_info = split_seq_info_syn_canonical(syn_gt)
    all_imgs, all_poses, all_k = [], [], []
    counts = [0]
    print("Preparing the synthetic training / val / test poses and images ...")
    for split in ['train', 'val', 'test']:
        if split == 'train':
            split_seq = train_info
        elif split == 'val':
            split_seq = val_info
        else:
            split_seq = test_info
        if sample_num is not None:
            split_seq = split_seq[-sample_num:]
        imgs, poses, ks = [], [], []
        for idx, one_info in enumerate(tqdm(split_seq)):
            image_id = one_info['image_id']
            one_k = np.array(syn_cam[str(image_id)]['cam_K']).reshape(3, 3)
            fname = os.path.join(data_root, 'train', str(cfg.data.seq_id).zfill(6), 'rgb', str(image_id).zfill(6) + ".png")
            one_img = imageio.imread(fname)
            mask_fname = os.path.join(data_root, 'train', str(cfg.data.seq_id).zfill(6), 'mask', str(image_id).zfill(6) + "_000000.png")
            mask_img = imageio.imread(mask_fname)
            mask_img = mask_img / mask_img.max()
            one_img = apply_mask_on_img(one_img, mask_img)
            # form obj pose
            one_obj_pose_R = np.array(syn_gt[str(image_id)][0]['cam_R_m2c']).reshape(3, 3)
            one_obj_pose_t = np.array(syn_gt[str(image_id)][0]['cam_t_m2c']).reshape(3)
            one_obj_pose = np.eye(4)
            one_obj_pose[:3, :3] = one_obj_pose_R
            one_obj_pose[:3, 3] = one_obj_pose_t
            # get_projected_points(one_obj_pose[:3, :4], one_k, obj_m, one_img=one_img, post_str="canonical_debug" + str(idx))
            # from object pose to camera pose, [R, t] -> [R^-1, -R^-1t]
            cam_pose = one_obj_pose.copy()
            cam_pose[:3, :3] = one_obj_pose_R.transpose()
            cam_pose[:3, -1] = -one_obj_pose_R.transpose() @ one_obj_pose_t
            # opencv to opengl
            diag = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32)).cpu().numpy()
            cam_pose = cam_pose @ diag
            ks.append(one_k)
            poses.append(cam_pose)
            imgs.append(one_img)
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        ks = np.array(ks).astype(np.float32)
        counts.append(counts[-1] + poses.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_k.append(ks)
    print("Finished preparing the training / val / test poses and images !")
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    images = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    ks = np.concatenate(all_k, 0)
    render_poses = torch.stack([pose_spherical(angle, 90.0, 400.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    # render_poses = gen_rotational_trajs(args, poses)
    
    i_train, i_val, i_test = i_split
    near, far = 0., 6.

    # Cast intrinsics to right types
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))
    render_poses = render_poses[...,:4]
    near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0.02)
    far *= 5
    data_dict = dict(HW=HW, Ks=ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=torch.tensor(poses), render_poses=torch.tensor(render_poses),
        images=torch.tensor(images), depths=None,
        irregular_shape=irregular_shape,
    )
    return data_dict


def load_synthetic_data_global(args, cfg, sample_num=None):
    """
    Load synthetic data in global coordinate system.
    """
    data_root = cfg.data.datadir
    seq_name = cfg.data.seq_name
    # load images and gts from deep im and rnnpose
    deepim_info_path = os.path.join(data_root, 'data_info/deepim', 'linemod_orig_deepim.info.train')
    eval_info_path = os.path.join(data_root, 'data_info', 'linemod_posecnn.info.eval')
    with open(deepim_info_path, 'rb') as f:
        seq_info = pickle.load(f)[seq_name]
    # load model
    ply_path = os.path.join(data_root, 'models', seq_name, seq_name + ".xyz")
    obj_m = o3d.io.read_point_cloud(ply_path, format='xyz')
    obj_m = np.asarray(obj_m.points)
    train_info, val_info, test_info = split_seq_info_global(seq_info)
    all_imgs, all_poses, all_k = [], [], []
    counts = [0]
    print("Preparing the synthetic training / val / test poses and images ...")
    for split in ['train', 'val', 'test']:
        if split == 'train':
            split_seq = train_info
        elif split == 'val':
            split_seq = val_info
        else:
            split_seq = test_info
        if sample_num is not None and sample_num > -1:
            split_seq = split_seq[-sample_num:]
        imgs, poses, ks = [], [], []
        for idx, one_info in enumerate(tqdm(split_seq)):
            image_id = one_info['index']
            one_k = np.array(one_info['K']).reshape(3, 3)
            fname = os.path.join(data_root, 'LM6d_converted/LM6d_refine', one_info['rgb_noisy_rendered'])
            one_img = imageio.imread(fname)
            # mask_fname = fname.replace('color', 'label')
            # mask_img = imageio.imread(mask_fname)
            # mask_img = mask_img / mask_img.max()
            one_img = change_background_from_black_to_white(one_img)
            # form obj pose
            pose_m = one_info['pose_noisy_rendered']
            one_obj_pose_R = np.array(pose_m[:3, :3]).reshape(3, 3)
            one_obj_pose_t = np.array(pose_m[:3, -1]).reshape(3)
            one_obj_pose = np.eye(4)
            one_obj_pose[:3, :3] = one_obj_pose_R
            one_obj_pose[:3, 3] = one_obj_pose_t
            # get_projected_points(pose_m, one_k, obj_m, one_img=one_img, post_str="global_debug" + str(idx))
            # opencv to opengl
            diag = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32)).cpu().numpy()
            cam_pose = one_obj_pose @ diag
            ks.append(one_k)
            poses.append(cam_pose)
            imgs.append(one_img)
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        ks = np.array(ks).astype(np.float32)
        counts.append(counts[-1] + poses.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_k.append(ks)
    print("Finished preparing the training / val / test poses and images !")
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    images = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    ks = np.concatenate(all_k, 0)
    # render_poses is not useful here.
    render_poses = torch.stack([pose_spherical(angle, 90.0, 400.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    
    i_train, i_val, i_test = i_split
    near, far = 0., 6.

    # Cast intrinsics to right types
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))
    render_poses = render_poses[...,:4]
    near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0.02)
    far *= 3
    data_dict = dict(HW=HW, Ks=ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=torch.tensor(poses), render_poses=torch.tensor(render_poses),
        images=torch.tensor(images), depths=None,
        irregular_shape=irregular_shape,
    )
    return data_dict


def load_linemod_data(args, cfg, vis_final=False, load_canonical=True):
    """
    Major loading function.
    """
    data_root = cfg.data.datadir
    seq_name = cfg.data.seq_name
    if args.program == 'tune_pose':
        # load images and gts from deep im and rnnpose
        deepim_info_path = os.path.join(data_root, 'data_info/deepim', 'linemod_orig_deepim.info.train')
        info_path = os.path.join(data_root, 'data_info', 'linemod_posecnn.info.eval')
        with open(info_path, 'rb') as f:
            seq_info = pickle.load(f)[seq_name]

        # collect pre-selected test indexs
        test_indexs = []
        for one_info in seq_info:
            index = one_info['index']
            if index not in test_indexs:
                test_indexs.append(index)
        
        # load model
        # xyz_model_p = os.path.join(data_root, 'models', seq_name, seq_name + ".xyz")
        # obj_m = o3d.io.read_point_cloud(xyz_model_p, format='xyz')
        # obj_m = np.asarray(obj_m.points)
        ply_model_p = os.path.join(data_root, 'models', seq_name, seq_name + ".ply")
        obj_m = np.array(o3d.io.read_point_cloud(ply_model_p, format='ply').points)
        obj_bb8 = get_bb8_of_model(obj_m)
        
        # load some pose initializations, not used actually
        posecnn_results_p = os.path.join(data_root, 'init_poses/linemod_posecnn_results.pkl')
        with open(posecnn_results_p, 'rb') as f:
            posecnn_results=pickle.load(f)[seq_name]
        test_pose_path = os.path.join(data_root, 'init_poses/pvnet/pvnet_linemod_test.npy')
        pvnet_results=np.load(test_pose_path, allow_pickle=True).flat[0][seq_name]

        # load testing info and gt
        test_gt_p = os.path.join(data_root, 'test', str(cfg.data.seq_id).zfill(6), 'scene_gt.json')
        test_gt = json.load(open(test_gt_p))
        
        # load camera K
        test_cam_p = os.path.join(data_root, 'test', str(cfg.data.seq_id).zfill(6), 'scene_camera.json')
        test_cam = json.load(open(test_cam_p))
        
        # summarize annotations
        sum_results = {}
        images = {}
        if args.sample_num > 0:
            seq_info = seq_info[:args.sample_num]
        print("Loading LineMod gts ...")
        for res in tqdm(seq_info):
            img_id = res['index']
            # fname = os.path.join(data_root, 'test', str(cfg.data.seq_id).zfill(6), 'rgb', str(img_id).zfill(6) + ".png")
            observed_p = res['rgb_observed_path']
            fname = os.path.join(data_root, 'LM6d_converted/LM6d_refine', observed_p)
            one_img = imageio.imread(fname)
            gt_pose, cam_k, posecnn_pose = res['gt_pose'], res['K'], res['pose_noisy_rendered']
            if vis_final:
                gt_2d = get_projected_points(gt_pose, cam_k, obj_m, one_img=one_img, post_str="init_gt")
                gt_2d = get_projected_points(posecnn_pose, cam_k, obj_m, one_img=one_img, post_str="init_posecnn")
                print("Visualization is generated!")
            images[int(img_id)] = torch.tensor(one_img.astype(np.float32))
            # imageio.imwrite("test.png", (one_img).astype(np.uint8))
            # quat, t = pose_to_blender(gt_pose)
        data_dict = dict(gts=seq_info, images=images, obj_m=obj_m, obj_bb8=obj_bb8)
        syn_data_dict = load_synthetic_data_canonical(args, cfg, sample_num=100)
        for key in syn_data_dict:
            if key not in data_dict:
                data_dict[key] = syn_data_dict[key]
        data_dict['syn_images'] = syn_data_dict['images']
        return data_dict
    elif load_canonical:  # during NeRF training, load synthetic infos
        data_dict = load_synthetic_data_canonical(args, cfg, args.sample_num)
        return data_dict
    else:
        data_dict = load_synthetic_data_global(args, cfg, args.sample_num)
        return data_dict
