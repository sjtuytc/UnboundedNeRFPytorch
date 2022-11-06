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
    # assert all_equal(train_HW), "image shapes are not all the same."
    test_HW = [train_HW[0] for i in range(test_num)]
    # assert all_equal(tr_K), "Ks are not all the same."
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


def gen_rotational_trajs(args, cfg, metadata, tr_c2w, train_HW, tr_K, rotate_angle=9):
    # We assume the metadata has been sorted here. 
    # Assume the first one is the center image.
    start_c2w, end_c2w = np.array(tr_c2w[0]), np.array(tr_c2w[-1])
    start_rot, end_rot = start_c2w[:3, :3], end_c2w[:3, :3]
    # forward to see the turning effect
    base_rot = R.from_matrix(start_rot)
    # base_rot = R.from_matrix(end_rot)
    base_pos = start_c2w[:3,3]
    # base_pos = end_c2w[:3,3]
    # generate rotating matries
    # rotate_interval = rotate_angle / test_num
    if args.program == 'tune_pose':
        test_num = 4
        rotate_interval = 10
    else:
        test_num = 5
        rotate_interval = 6
        # test_num = 200
        # rotate_interval = 0.1
    forward_dis_max = 0.0  # default is 0.0
    all_rot_yzx = [base_rot.as_euler('yzx', degrees=True)] 
    for i in range(test_num - 1):
        if all_rot_yzx:
            prev_rot = all_rot_yzx[-1]
        else:
            prev_rot = base_rot.as_euler('yzx', degrees=True)
        new_rot = [prev_rot[0], prev_rot[1] + rotate_interval, prev_rot[2]]
        all_rot_yzx.append(new_rot)
    all_rot = [R.from_euler('yzx', rot, degrees=True).as_matrix() for rot in all_rot_yzx]
    all_c2ws = [start_c2w.copy() for i in range(test_num)]  # initialize
    for i, c2w in enumerate(all_c2ws):
        all_c2ws[i][:3, :3] = all_rot[i]
        # forward_dis = (1 - np.cos(i / len(all_c2ws) * np.pi / 2)) * forward_dis_max
        forward_dis = forward_dis_max
        cur_pos = [base_pos[0] - forward_dis, base_pos[1], base_pos[2]]
        all_c2ws[i][:3, 3] = cur_pos
    assert train_HW[0] == train_HW[-1], "image shapes are not the same for the first and the last frame."
    test_HW = [train_HW[0] for i in range(test_num)]
    test_K = [tr_K[0] for i in range(test_num)]
    return all_c2ws, test_HW, test_K