import pdb, os, cv2
from turtle import onkey
from random import sample
import imageio
from FourierGrid import utils
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from FourierGrid import utils, dvgo, dcvgo, dmpigo
from FourierGrid.bbox_compute import compute_bbox_by_cam_frustrm
import shutil


def select_k_nearest_points(idx, positions, k):
    positions = np.array(positions)
    distances = [np.linalg.norm(a-positions[idx]) for a in positions]
    sorted_idxs = sorted(zip(range(len(distances)), distances), key=lambda row: row[1])
    sorted_idxs = [i for i, j in sorted_idxs[:1+k]]  # the first one is itself
    return sorted_idxs


def move_idxs_to_folder(data_dict, sampled_idxs, save_path, data_root="data/sep19_ordered_dataset"):
    images = data_dict['images']
    rgb_paths = [images[idx] for idx in sampled_idxs]
    full_paths = [os.path.join(data_root, path) for path in rgb_paths]
    for one_p in full_paths:
        image_name = one_p.split("/")[-1]
        shutil.copyfile(one_p, os.path.join(save_path, image_name))
    return


def render_idxs(data_dict, sampled_idxs, save_path, data_root="data/sep19_ordered_dataset", fps=15, output_res=(800, 608)):
    images = data_dict['images']
    rgb_paths = [images[idx] for idx in sampled_idxs]
    rgb_images = [imageio.imread(os.path.join(data_root, path)) / 255. for path in rgb_paths]
    rgb_images = [cv2.resize(img, output_res) for img in rgb_images]
    shapes = [img.shape for img in rgb_images]
    imageio.mimwrite(save_path, utils.to8b(rgb_images), fps=fps, quality=8)
    print(f"Demo video is saved at {save_path}.")
    return


def get_rotation_kp_2d(args, cfg, data_dict, sample_idxs):
    # get rotation vector in 2D planes
    poses, HW, Ks, near, far = data_dict['poses'][sample_idxs], data_dict['HW'][sample_idxs], data_dict['Ks'][sample_idxs], data_dict['near'], data_dict['far']
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, HW=HW, Ks=Ks, poses=poses, i_train=list(range(len(poses))), near=near, far=far, near_clip=data_dict['near_clip'])
    near, far = data_dict['near'], data_dict['far']
    if data_dict['near_clip'] is not None:
        near = data_dict['near_clip']
    rotations = []
    for c2w, (H, W), K in zip(poses, HW, Ks):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
        # cam_o = rays_o[0, 0].cpu().numpy()
        # cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
        cam_d = rays_d[rays_d.shape[0]//2, rays_d.shape[1]//2].cpu().numpy()
        rotations.append(cam_d)
    return rotations
    
    
def run_export_bbox_cams(args, cfg, data_dict, sample_idxs, save_path): 
    # save the sampled camera in order to visualize it
    print('Export bbox and cameras...')
    poses, HW, Ks, near, far = data_dict['poses'][sample_idxs], data_dict['HW'][sample_idxs], data_dict['Ks'][sample_idxs], data_dict['near'], data_dict['far']
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, HW=HW, Ks=Ks, poses=poses, i_train=list(range(len(poses))), near=near, far=far, near_clip=data_dict['near_clip'])
    near, far = data_dict['near'], data_dict['far']
    if data_dict['near_clip'] is not None:
        near = data_dict['near_clip']
    cam_lst = []
    for c2w, (H, W), K in zip(poses, HW, Ks):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
        cam_o = rays_o[0, 0].cpu().numpy()
        cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
        cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
    dir_name = os.path.dirname(save_path)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path,
        xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
        cam_lst=np.array(cam_lst))
    print(f"The cam path has been saved at {save_path}.")


def run_gen_cam_paths(args, cfg, data_dict, core_cam=None, straight_length=100):
    print("Generating camera paths ...")
    # retrieve set of image lists for rendering videos
    images = data_dict['images']
    val_imgs = images[data_dict['i_val'][0]:data_dict['i_val'][-1]]
    whole_cam_idxs = data_dict['cam_idxs']
    whole_positions = data_dict['positions']
    whole_poses = data_dict['poses']
    # generate straight videos
    idxs_all = list(range(len(data_dict['positions'])))
    if core_cam is None:  # used as the core camera for video rendering
        core_cam = max(set(whole_cam_idxs), key=whole_cam_idxs.count)
    core_idxs = [idx for idx in idxs_all if whole_cam_idxs[idx] == core_cam]
    sampled_positions = [whole_positions[idx] for idx in core_idxs]
    sorted_idxs = sorted(zip(core_idxs, sampled_positions), key=lambda row: (row[1][1], row[1][0]))
    sorted_idxs = [i for i, j in sorted_idxs]
    sample_start = len(sorted_idxs) // 2 - straight_length // 2  # sample from mid
    straight_idxs = sorted_idxs[sample_start: sample_start + straight_length]
    save_p = 'data/samples/demo_video'
    os.makedirs(save_p, exist_ok=True)
    # render_idxs(data_dict, straight_idxs, save_path=os.path.join(save_p, 'straight.mp4'))
    # run_export_bbox_cams(args, cfg, data_dict=data_dict, sample_idxs=straight_idxs, save_path=os.path.join(save_p, 'straight_cam.npz'))

    # visualizing cameras in different rotations
    close_idxs = select_k_nearest_points(sample_start, whole_positions, k=15)
    rotations = get_rotation_kp_2d(args, cfg, data_dict, close_idxs)
    sorted_idxs = sorted(zip(close_idxs, rotations), key=lambda row: (row[1][1], row[1][0]))
    sorted_idxs = [i for i, j in sorted_idxs]
    cam2idxs = {}
    # save sorted indexes in to the disk.
    save_idxs = []
    for one_idx in sorted_idxs:
        cam_idx = whole_cam_idxs[one_idx]
        if cam_idx not in cam2idxs:
            cam2idxs[cam_idx] = [one_idx] + straight_idxs
            print(f'cam_id:{cam_idx}, image path: {images[one_idx]}, original idx: {one_idx}.')
            save_idxs.append(one_idx)
            run_export_bbox_cams(args, cfg, data_dict=data_dict, sample_idxs=cam2idxs[cam_idx], save_path=os.path.join(save_p, f'cam_{cam_idx}.npz'))
    # move_idxs_to_folder(data_dict, save_idxs, save_path=save_p)

    # close_poses = [whole_poses[idx] for idx in close_idxs]
    # rotations = [R.from_matrix(pose[:3, :3]) for pose in close_poses]
    # rot_degrees = [r.as_euler('zxy', degrees=True) for r in rotations]
    # sorted_idxs = sorted(zip(close_idxs, rot_degrees), key=lambda row: row[1][0])
    # sorted_idxs = [i for i, j in sorted_idxs]
    final_rot_idxs = sorted_idxs
    
    # start_pos_in_sort = sorted_idxs.index(sample_start)
    # if start_pos_in_sort > len(sorted_idxs) // 2:
    #     final_rot_idxs = sorted_idxs[:start_pos_in_sort]  # later
    # else:
    #     final_rot_idxs = sorted_idxs[start_pos_in_sort:]
    combined_idxs = final_rot_idxs + straight_idxs
    render_idxs(data_dict, combined_idxs, save_path=os.path.join(save_p, 'rot.mp4'))
    run_export_bbox_cams(args, cfg, data_dict=data_dict, sample_idxs=combined_idxs, save_path=os.path.join(save_p, 'rot_cam.npz'))
