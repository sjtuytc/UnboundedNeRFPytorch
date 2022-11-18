import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R


def chordal_distance(R1,R2):
    return np.sqrt(np.sum((R1-R2)*(R1-R2))) 


def rotation_angle_chordal(R1, R2):
    return 2*np.arcsin(chordal_distance(R1,R2)/np.sqrt(8))


def cal_pose_rot_diff(pose1, pose2):
    ang_err_chordal = rotation_angle_chordal(pose1[:3, :3], pose2[:3, :3])
    return ang_err_chordal


def rot_diff_to_norm_angle(rotation_difference):
    theta = np.arccos((np.trace(rotation_difference) - 1) / 2)
    theta = np.rad2deg(theta)
    euler = R.from_matrix(rotation_difference).as_euler('zyx', degrees=True)
    norm_angle = np.linalg.norm(euler)
    return norm_angle


def cal_one_add(model_points, pose_pred, pose_targets, syn=False):
    model_pred = np.dot(model_points, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_targets = np.dot(model_points, pose_targets[:, :3].T) + pose_targets[:, 3]
    if syn:
        from thirdparty.nn import nn_utils  # TODO: solve this reference
        idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
        # idxs = find_nearest_point_idx(model_pred, model_targets)
        mean_dist = np.mean(np.linalg.norm(
            model_pred[idxs] - model_targets, 2, 1))
    else:
        mean_dist = np.mean(np.linalg.norm(
            model_pred - model_targets, axis=-1))
    return mean_dist


def pose_rot_interpolation(pose_a, pose_b, inter_num=100):
    '''
    interpolate poses ASSUMING pose = [R, Rt] (in the canonical coordinate system). 
    '''
    pose_a, pose_b = pose_a.cpu().numpy(), pose_b.cpu().numpy()
    pose_a_rot, pose_b_rot = pose_a[:3, :3], pose_b[:3, :3]
    pose_a_trans, pose_b_trans = pose_a[:3, -1], pose_b[:3, -1]
    inv_pose_a_rot, inv_pose_b_rot = np.linalg.inv(pose_a_rot), np.linalg.inv(pose_b_rot)
    # inv_pose_a_rot@pose_a_trans is equal to inv_pose_b_rot@pose_b_trans, be [0, 0, 400] by default
    ori_t = (inv_pose_a_rot@pose_a_trans).astype(np.int)
    # generate rotations
    rotation_a, rotation_b = R.from_matrix(pose_a_rot), R.from_matrix(pose_b_rot)
    euler_order = 'xyz'
    rotation_a_euler, rotation_b_euler = rotation_a.as_euler(euler_order), rotation_b.as_euler(euler_order)
    all_rotations_euler = [rotation_a_euler + i / inter_num * (rotation_b_euler - rotation_a_euler) for i in range(inter_num)]
    all_rotations = [R.from_euler(euler_order, rot_euler) for rot_euler in all_rotations_euler]
    all_rotations = [rot.as_matrix() for rot in all_rotations]
    # form pose matrixs
    poses = [pose_a.copy() for i in range(inter_num)]
    for i in range(inter_num):
        poses[i][:3, :3] = all_rotations[i]
        poses[i][:3, -1] = poses[i][:3, :3] @ ori_t
    return poses


def pose_to_blender(pose):
    rot = pose[:3, :3]
    quat = R.from_matrix(rot).as_quat()
    trans = pose[:3, -1]
    inv_rot = np.linalg.inv(rot)
    cam_loc = - inv_rot @ trans 
    return quat, cam_loc