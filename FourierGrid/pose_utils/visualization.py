import os
import cv2
import pdb
import torch
import imageio
import numpy as np


def visualize_2d_points(points_2d, bg_image, save_root=None, pre_str="", post_str=""):
    """
    points_2d: [N, 2] denotes the points in the 2D space
    bg_image: background image for visualization
    post_str: adding some description
    """
    vis_img = np.zeros(bg_image.shape).astype(np.uint8)
    points_2d = points_2d.astype(np.int)
    vis_img[points_2d[:, -1], points_2d[:, 0], :] = 255
    if save_root is None:
        imageio.imwrite(f'{pre_str}ori{post_str}.png', bg_image.astype(np.uint8))
        imageio.imwrite(f'{pre_str}projected{post_str}.png', vis_img.astype(np.uint8))
        imageio.imwrite(f'{pre_str}composed{post_str}.png', np.maximum(bg_image, vis_img).astype(np.uint8))
    else:
        imageio.imwrite(os.path.join(save_root, f'{pre_str}ori{post_str}.png'), bg_image.astype(np.uint8))
        imageio.imwrite(os.path.join(save_root, f'{pre_str}projected{post_str}.png'), vis_img.astype(np.uint8))
        imageio.imwrite(os.path.join(save_root, f'{pre_str}composed{post_str}.png'), np.maximum(bg_image, vis_img).astype(np.uint8))
    return


def get_projected_points(cam_pose, cam_k, obj_m, one_img=None, save_root=None, pre_str="", post_str=""):
    from FourierGrid.pose_utils.projection import get_projected_points
    return get_projected_points(cam_pose, cam_k, obj_m, one_img, save_root, pre_str, post_str)


def draw_bbox_8_2D(draw_img, bbox_8_2D, color = (0, 255, 0), thickness = 2):
    """ Draws the 2D projection of a 3D model's cuboid on an image with a given color.
    # Arguments
        draw_img     : The image to draw on.
        bbox_8_2D    : A [8 or 9, 2] matrix containing the 8 corner points (x, y) and maybe also the centerpoint.
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    #convert bbox to int and tuple
    bbox = np.copy(bbox_8_2D).astype(np.int32)
    bbox = tuple(map(tuple, bbox))
    #lower level
    cv2.line(draw_img, bbox[0], bbox[1], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[2], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[3], color, thickness)
    cv2.line(draw_img, bbox[0], bbox[3], color, thickness)
    #upper level
    cv2.line(draw_img, bbox[4], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[5], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[6], bbox[7], color, thickness)
    cv2.line(draw_img, bbox[4], bbox[7], color, thickness)
    #sides
    cv2.line(draw_img, bbox[0], bbox[4], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[3], bbox[7], color, thickness)
    
    #check if centerpoint is also available to draw
    if len(bbox) == 9:
        #draw centerpoint
        cv2.circle(draw_img, bbox[8], 3, color, -1)
    

def visualize_pose_prediction(pose_a, pose_b, cam_k, obj_bb8, bg_img, save_root=None, a_color=(170, 214, 85), b_color=(66, 51, 122), pre_str='', post_str=''):
    # get projected points only so save_root=None, post_str=""
    new_bg_img = bg_img.copy()
    bb8_2d_a = get_projected_points(pose_a, cam_k, obj_bb8, one_img=None, save_root=None, pre_str="", post_str="")
    bb8_2d_b = get_projected_points(pose_b, cam_k, obj_bb8, one_img=None, save_root=None, pre_str="", post_str="")
    draw_bbox_8_2D(new_bg_img, bb8_2d_a, color=a_color, thickness=2)
    draw_bbox_8_2D(new_bg_img, bb8_2d_b, color=b_color, thickness=2)
    save_path = f'{pre_str}compare_pose{post_str}.png'
    if save_root is not None:
        save_path = os.path.join(save_root, save_path)
    imageio.imwrite(save_path, new_bg_img.astype(np.uint8))
