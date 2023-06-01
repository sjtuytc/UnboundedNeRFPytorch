import click
import os
import numpy as np
import cv2 as cv
from os.path import join as pjoin
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def inter_two_poses(pose_a, pose_b, alpha):
    ret = np.zeros([3, 4], dtype=np.float64)
    rot_a = R.from_matrix(pose_a[:3, :3])
    rot_b = R.from_matrix(pose_b[:3, :3])
    key_rots = R.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], 0))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    rot = slerp(1. - alpha)
    ret[:3, :3] = rot.as_matrix()
    ret[:3, 3] = (pose_a * alpha + pose_b * (1. - alpha))[:3, 3]
    return ret


def inter_poses(key_poses, n_out_poses, sigma=1.):
    n_key_poses = len(key_poses)
    out_poses = []
    for i in range(n_out_poses):
        w = np.linspace(0, n_key_poses - 1, n_key_poses)
        w = np.exp(-(np.abs(i / n_out_poses * n_key_poses - w) / sigma)**2)
        w = w + 1e-6
        w /= np.sum(w)
        cur_pose = key_poses[0]
        cur_w = w[0]
        for j in range(0, n_key_poses - 1):
            cur_pose = inter_two_poses(cur_pose, key_poses[j + 1], cur_w / (cur_w + w[j + 1]))
            cur_w += w[j + 1]

        out_poses.append(cur_pose)

    return np.stack(out_poses)

@click.command()
@click.option('--data_dir', type=str)
@click.option('--key_poses', type=str)
@click.option('--n_out_poses', type=int, default=240)
def hello(data_dir, n_out_poses, key_poses):
    poses = np.load(pjoin(data_dir, 'cams_meta.npy')).reshape(-1, 27)[:, :12].reshape(-1, 3, 4)
    n_poses = len(poses)
    key_poses = np.array([int(_) for _ in key_poses.split(',')])
    key_poses = poses[key_poses]

    out_poses = inter_poses(key_poses, n_out_poses)
    out_poses = np.ascontiguousarray(out_poses.astype(np.float64))
    np.save(pjoin(data_dir, 'poses_render.npy'), out_poses)


if __name__ == '__main__':
    hello()
