import numpy as np
from scipy.spatial.transform import Rotation as R
import pdb
from itertools import groupby


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def rotate_rot_matrix_by_degree(rot, rot_degree, axis='y'):
    rotate_r = R.from_euler(axis, -rot_degree, degrees=True)
    rot_matrix_new = np.matmul(rot, rotate_r.as_matrix())
    return rot_matrix_new


def gen_dummy_trajs(metadata, tr_c2w, train_HW, tr_K, test_num=100):
    assert all_equal(train_HW), "image shapes are not all the same."
    test_HW = [train_HW[0] for i in range(test_num)]
    assert all_equal(tr_K), "Ks are not all the same."
    test_K = [tr_K[0] for i in range(test_num)]
    all_c2ws = tr_c2w.copy()[:test_num]  # initialize
    all_c2ws = [np.array(c2w) for c2w in all_c2ws]
    return all_c2ws, test_HW, test_K


def gen_straight_trajs(metadata, tr_c2w, train_HW, tr_K, tr_cam_idx, train_pos, test_num=100, rotate_angle=2, rot_freq=20):
    assert all_equal(train_HW), "image shapes are not all the same."
    test_HW = [train_HW[0] for i in range(test_num)]
    assert all_equal(tr_K), "Ks are not all the same."
    test_K = [tr_K[0] for i in range(test_num)]
    assert all_equal(tr_cam_idx), "Cameras are not all the same."
    test_cam_idxs = [tr_cam_idx[0] for i in range(test_num)]
    all_c2ws = tr_c2w.copy()[:test_num]  # initialize
    all_c2ws = [np.array(c2w) for c2w in all_c2ws]
    average_z = np.mean([c2w[2, 3] for c2w in all_c2ws])
    for i, c2w in enumerate(all_c2ws):
        final_rot = rotate_angle * np.sin(i / rot_freq * 2 * np.pi)
        all_c2ws[i][:3, :3] = rotate_rot_matrix_by_degree(all_c2ws[i][:3, :3], final_rot, axis='y')
    return all_c2ws, test_HW, test_K, test_cam_idxs
