'''
Modify from
https://github.com/Kai-46/nerfplusplus/blob/master/data_loader_split.py
'''
import os
import glob
import scipy
import imageio
import numpy as np
import torch

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


def load_data_split(split_dir, skip=1, try_load_min_depth=True, only_img_files=False):

    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    if only_img_files:
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        return img_files

    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])

    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        img_files = img_files[::skip]
        assert(len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt

    # mask files
    mask_files = find_files('{}/mask'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(mask_files) > 0:
        mask_files = mask_files[::skip]
        assert(len(mask_files) == cam_cnt)
    else:
        mask_files = [None, ] * cam_cnt

    # min depth files
    mindepth_files = find_files('{}/min_depth'.format(split_dir), exts=['*.png', '*.jpg'])
    if try_load_min_depth and len(mindepth_files) > 0:
        mindepth_files = mindepth_files[::skip]
        assert(len(mindepth_files) == cam_cnt)
    else:
        mindepth_files = [None, ] * cam_cnt

    return intrinsics_files, pose_files, img_files, mask_files, mindepth_files


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


def load_nerfpp_data(basedir, rerotate=True):
    tr_K, tr_c2w, tr_im_path = load_data_split(os.path.join(basedir, 'train'))[:3]
    te_K, te_c2w, te_im_path = load_data_split(os.path.join(basedir, 'test'))[:3]
    assert len(tr_K) == len(tr_c2w) and len(tr_K) == len(tr_im_path)
    assert len(te_K) == len(te_c2w) and len(te_K) == len(te_im_path)

    # Determine split id list
    i_split = [[], []]
    i = 0
    for _ in tr_c2w:
        i_split[0].append(i)
        i += 1
    for _ in te_c2w:
        i_split[1].append(i)
        i += 1

    # Load camera intrinsics. Assume all images share a intrinsic.
    K_flatten = np.loadtxt(tr_K[0])
    for path in tr_K:
        assert np.allclose(np.loadtxt(path), K_flatten)
    for path in te_K:
        assert np.allclose(np.loadtxt(path), K_flatten)
    K = K_flatten.reshape(4,4)[:3,:3]

    # Load camera poses
    poses = []
    for path in tr_c2w:
        poses.append(np.loadtxt(path).reshape(4,4))
    for path in te_c2w:
        poses.append(np.loadtxt(path).reshape(4,4))

    # Load images
    imgs = []
    for path in tr_im_path:
        imgs.append(imageio.imread(path) / 255.)
    for path in te_im_path:
        imgs.append(imageio.imread(path) / 255.)

    # Bundle all data
    imgs = np.stack(imgs, 0)
    poses = np.stack(poses, 0)
    i_split.append(i_split[1])
    H, W = imgs.shape[1:3]
    focal = K[[0,1], [0,1]].mean()

    # Generate movie trajectory
    render_poses_path = sorted(glob.glob(os.path.join(basedir, 'camera_path', 'pose', '*txt')))
    render_poses = []
    for path in render_poses_path:
        render_poses.append(np.loadtxt(path).reshape(4,4))
    render_poses = np.array(render_poses)
    render_K = np.loadtxt(glob.glob(os.path.join(basedir, 'camera_path', 'intrinsics', '*txt'))[0]).reshape(4,4)[:3,:3]
    render_poses[:,:,0] *= K[0,0] / render_K[0,0]
    render_poses[:,:,1] *= K[1,1] / render_K[1,1]
    if rerotate:
        poses, render_poses = rerotate_poses(poses, render_poses)

    render_poses = torch.Tensor(render_poses)

    return imgs, poses, render_poses, [H, W, focal], K, i_split

