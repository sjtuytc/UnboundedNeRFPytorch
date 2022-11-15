import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R


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


def pose_sample(center_pose, sample_range=10, sample_num=100):
    """
    Sample poses around some point.
    """
    
    pdb.set_trace()

