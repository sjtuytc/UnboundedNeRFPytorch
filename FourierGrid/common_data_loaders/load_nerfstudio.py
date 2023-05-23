import numpy as np
import os, imageio
import torch
import scipy
import pdb
import json
from FourierGrid.tools.colmap_utils.pose_utils import load_colmap_data_nerfstudio

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

    from shutil import copy
    from subprocess import check_output

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
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['magick mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data_nerfstudio_more(basedir, poses, names, factor=None, width=None, height=None, load_imgs=True, load_depths=False):
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
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png')]
    assert len(imgfiles) > 1, "Images are too few. Do they exist?"

    if len(imgfiles) < 3:
        print('Too few images...')
        import sys; sys.exit()

    sh = imageio.imread(imgfiles[0]).shape
    if poses.shape[1] == 4:
        raise RuntimeError("The shape of pose does not contain hwf, this should not happen.")
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    if not load_imgs:
        return poses
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:,-1,0])
    if not load_depths:
        return poses, imgs

    depthdir = os.path.join(basedir, 'stereo', 'depth_maps')
    assert os.path.exists(depthdir), f'Dir not found: {depthdir}'

    depthfiles = [os.path.join(depthdir, f) for f in sorted(os.listdir(depthdir)) if f.endswith('.geometric.bin')]
    assert poses.shape[-1] == len(depthfiles), 'Mismatch between imgs {} and poses {} !!!!'.format(len(depthfiles), poses.shape[-1])

    depths = [depthread(f) for f in depthfiles]
    depths = np.stack(depths, -1)
    print('Loaded depth data', depths.shape)
    return poses, imgs, depths


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


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate)*zdelta, 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


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


def cal_colmap_bounds(poses, pts3d, sorted_names):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1] # N
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())
    
    bds = []
    for i in sorted_names:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth)
        bds.append(np.array([close_depth, inf_depth]))
    bds = np.stack(bds, axis=1)
    return bds
    

def load_nerfstudio_data(basedir, factor=8, width=None, height=None,
                   recenter=True, rerotate=True, bd_factor=.75, dvgohold=8,
                   spherify=False, path_zflat=False, load_depths=False,
                   movie_render_kwargs={}):
    poses, train_num, pts3d, sorted_names, names = load_colmap_data_nerfstudio(basedir)
    bds = cal_colmap_bounds(poses[:train_num], pts3d, sorted_names)
    poses, imgs, *depths = _load_data_nerfstudio_more(basedir, poses, names, factor=factor, width=width, height=height,
                                                           load_depths=load_depths) # factor=8 downsamples original imgs by 8x
    # poses: [3, 5, N], bds: [2, N], imgs: [H, W, 3, N], depths: []
    print('Loaded', basedir, bds.min(), bds.max())
    if load_depths:
        depths = depths[0]
    else:
        depths = 0
    # Correct rotation matrix ordering and move variable dim to axis 0. Commented out by Zelin.
    # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    if bds.min() < 0 and bd_factor is not None:
        print('Found negative z values from SfM sparse points!?')
        print('Please try bd_factor=None. This program is terminating now!')
        import sys; sys.exit()
    sc = 0.1 / poses[:,:3,3].max() if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    depths *= sc
    if recenter:  # not affected too much
        poses = recenter_poses(poses)

    if spherify:  # not affected too much
        poses, radius, bds, depths = spherify_poses(poses, bds, depths)
        if rerotate:
            poses = rerotate_poses(poses)
  
    render_poses = torch.Tensor(poses[4:])  # this is wrong, just for debug
    poses = poses[:train_num]
    # this part is written by DVGO, called dvgo split.
    if dvgohold > 0:
        c2w = poses_avg(poses)
        dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
        i_test = np.argpartition(dists, dvgohold).tolist()[:dvgohold]
        print('HOLDOUT views are', i_test)
    else:
        i_test = [0]
        print("DVGO hold is illegal, using [0] as test split!")
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    return images, depths, poses, bds, render_poses, i_test
