import torch
import numpy as np
from FourierGrid.pose_utils.visualization import *


def get_projected_points(cam_pose, cam_k, obj_m, one_img=None, save_root=None, pre_str="", post_str=""):
    point_num = obj_m.shape[0]
    homo_points_3d = np.concatenate([obj_m, np.ones((point_num, 1))], axis=-1)
    batch_cam_pose = torch.tensor(cam_pose).unsqueeze(0).repeat(point_num, 1, 1)
    batch_cam_k = torch.tensor(cam_k).unsqueeze(0).repeat(point_num, 1, 1)
    homo_points_2d = torch.bmm(batch_cam_pose, torch.tensor(homo_points_3d).unsqueeze(-1))
    homo_points_2d = torch.bmm(batch_cam_k, homo_points_2d)
    points_2d = homo_points_2d.squeeze()
    points_2d = points_2d[:, :2] / points_2d[:, -1].unsqueeze(-1).repeat(1, 2)
    points_2d = points_2d.cpu().numpy()
    if one_img is not None:  # for visualization:
        visualize_2d_points(points_2d=points_2d, bg_image=one_img, save_root=save_root, pre_str=pre_str, post_str=post_str)
    return points_2d
