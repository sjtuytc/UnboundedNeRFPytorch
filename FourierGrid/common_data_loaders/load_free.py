import numpy as np
import os, imageio
import torch
import scipy
import cv2
import pdb
from shutil import copy
from subprocess import check_output
from FourierGrid.trajectory_generators.interp_traj import *

    
########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original
def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def depthread(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'jpeg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            print("Image folder exists, do not call the resize function.")
            continue
        os.makedirs(imgdir, exist_ok=True)
        print('Minifying', r, basedir)
        # check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        ext = imgs[0].split('.')[-1]
        for idx, one_img_p in enumerate(imgs):
            one_img = cv2.imread(one_img_p)
            ori_h, ori_w = one_img.shape[0], one_img.shape[1]
            if isinstance(r, int):
                target_h, target_w = int(ori_h / r), int(ori_w / r)
            else:
                target_h, target_w = r[0], r[1]
            resized = cv2.resize(one_img, (target_w, target_h), interpolation = cv2.INTER_AREA)
            target_img_p = one_img_p.replace(imgdir_orig, imgdir)
            cv2.imwrite(target_img_p, resized)
        # args = ' '.join(['convert mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        # print(args)
        # os.chdir(imgdir)
        # check_output(args, shell=True)
        # os.chdir(wd)

        # if ext != 'png':
        #     check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
        #     print('Removed duplicates')
        print('Done')


def normalize_scene(poses, n_images, bounds):
    # TODO: vdalidate the effectiveness of this function
    cam_pos = poses[:, :, 3].clone()
    center_ = cam_pos.mean(dim=0, keepdim=False)
    bias = cam_pos - center_.unsqueeze(0)
    radius_ = torch.linalg.norm(bias, ord=2, dim=-1, keepdim=False).max().item()
    cam_pos = (cam_pos - center_.unsqueeze(0)) / radius_
    poses[:, :, 3] = cam_pos
    bounds = (bounds / radius_)
    return poses, bounds, center_, radius_


def load_images_from_disk(basedir, factor, height, width):
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    sfx = ''
    if height is not None and width is not None:
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif factor is not None and factor != 1:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    imgdir = os.path.join(basedir, 'images' + sfx)
    print(f'Loading images from {imgdir}')
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning' )
        import sys; sys.exit()
        return
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if len(imgfiles) < 3:
        print('Too few images...')
        import sys; sys.exit()

    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, 0)
    return imgs, factor

def normalize(x):
    return x / np.linalg.norm(x)

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate)*zdelta, 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses, render_poses):
    poses_ = poses + 0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    
    # apply c2w to render poses
    render_poses_ = render_poses + 0
    bottom = np.reshape([0,0,0,1.], [1,4])
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [render_poses.shape[0],1,1])
    render_poses = np.concatenate([render_poses[:,:3,:4], bottom], -2)
    render_poses = np.linalg.inv(c2w) @ render_poses
    render_poses_[:, :3, :4] = render_poses[:, :3, :4]
    render_poses = render_poses_
    return poses, render_poses


def rerotate_poses(poses):
    poses = np.copy(poses)
    centroid = poses[:,:3,3].mean(0)

    poses[:,:3,3] = poses[:,:3,3] - centroid

    # Find the minimum pca vector with minimum eigen value
    x = poses[:,:,3]
    mu = x.mean(0)
    cov = np.cov((x-mu).T)
    ev , eig = np.linalg.eig(cov)
    cams_up = eig[:,np.argmin(ev)]
    if cams_up[1] < 0:
        cams_up = -cams_up

    # Find rotation matrix that align cams_up with [0,1,0]
    R = scipy.spatial.transform.Rotation.align_vectors(
            [[0,1,0]], cams_up[None])[0].as_matrix()

    # Apply rotation and add back the centroid position
    poses[:,:3,:3] = R @ poses[:,:3,:3]
    poses[:,:3,[3]] = R @ poses[:,:3,[3]]
    poses[:,:3,3] = poses[:,:3,3] + centroid
    return poses

#####################


def spherify_poses(poses, bds, depths):

    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    radius = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))

    sc = 1./radius
    poses_reset[:,:3,3] *= sc
    bds *= sc
    radius *= sc
    depths *= sc

    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)

    return poses_reset, radius, bds, depths


def load_free_data(args, basedir, factor=8, width=None, height=None,
                   recenter=True, rerotate=True,
                   bd_factor=.75, spherify=False, path_zflat=False, load_depths=False,
                   movie_render_kwargs={}, training_ids=None, generate_render_poses=True, n_out_poses=200):
    # 1. load and parse poses, images, and bounds
    meta_pose = torch.tensor(np.load(os.path.join(basedir, 'cams_meta.npy')))
    n_images = meta_pose.shape[0]
    cam_data = meta_pose.reshape(n_images, 27)
    poses = cam_data[:, 0:12].reshape(-1, 3, 4)
    intri = cam_data[:, 12:21].reshape(-1, 3, 3)
    poses = poses.cpu().numpy()
    intri = intri.cpu().numpy()
    
    # 2. Rotation matrix correct, this has been done in colmap2standard
    # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs, factor = load_images_from_disk(basedir, factor, height, width)
    intri[..., :2, :3] /= factor
    dist_params = cam_data[:, 21:25].reshape(-1, 4)
    bounds = cam_data[:, 25:27].reshape(-1, 2)

    # 2. normalize scenes
    # poses, bounds, center, radius = normalize_scene(poses, n_images, bounds)
    assert not load_depths, "do not support loading depths"
    assert len(imgs.shape) == 4, "image shape is not correct!"
    assert intri[0][0][0] == intri[1][0][0] and intri[1][0][0] == intri[2][0][0], "focal length are varying!"

    # filter by training_ids
    if training_ids is not None:
        poses = np.array([poses[id] for id in training_ids])
        intri = np.array([intri[id] for id in training_ids])
        imgs = np.array([imgs[id] for id in training_ids])
        bounds = bounds[training_ids]

    # 3. load render camera poses or generate render poses on the fly
    if generate_render_poses:
        key_poses_indexs = np.arange(0, poses.shape[0], 5)
        key_poses = poses[key_poses_indexs]
        render_poses_ = inter_poses(key_poses, n_out_poses)
    else:
        poses_render_path = os.path.join(basedir, "poses_render.npy")
        arr = np.load(poses_render_path)
        cam_data = torch.from_numpy(arr.astype(np.float64)).to(torch.float32).cuda()
        n_render_poses = arr.shape[0]
        cam_data = cam_data.reshape((-1, 3, 4))
        cam_data = cam_data[:n_render_poses, :, :]
        render_poses_ = cam_data.clone()  # [n, 3, 4]
        # render_poses_[:, :3, 3] = (render_poses_[:, :3, 3] - center.unsqueeze(0)) / radius  #commented out for debugging
        render_poses_ = render_poses_.cpu().numpy()
    hwf = np.array([[imgs.shape[1], imgs.shape[2], intri[0][0][0]]for _ in range(render_poses_.shape[0])])
    render_poses_ = np.concatenate((render_poses_, hwf.reshape((render_poses_.shape[0], 3, 1))), axis=2)
    
    # 4. relax bounds
    # bounds_factor = [0.5, 4.0]
    bounds = torch.stack([bounds[:, 0], bounds[:, 1]], dim=-1)
    bounds.clamp_(1e-2, 1e9)
    bounds = bounds.cpu().numpy()
    near = bounds.min().item()
    # sc = 1 / (near * bd_factor)  # 0.12 by default
    sc = 1.0  # 0.12 in M360
    poses[:,:3,3] *= sc
    render_poses_[:, :3, 3] *= sc
    hwf = np.array([[imgs.shape[1], imgs.shape[2], intri[0][0][0]]for _ in range(imgs.shape[0])])
    poses = np.concatenate((poses, hwf.reshape((imgs.shape[0], 3, 1))), axis=2)
    poses, render_poses_ = recenter_poses(poses, render_poses_)
    
    # 5. get test ID, this part is written by DVGO.
    # c2w = poses_avg(poses)
    # dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    # i_test = np.argmin(dists)
    # i_test = [i_test]
    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(imgs.shape[0])[::args.llffhold]
    return imgs, 0, intri, poses, bounds, render_poses_, i_test
