import numpy as np
import datetime, os
import cv2
import os
import pdb
import glob
import tensorflow as tf
import torch
import glob
import os
import numpy as np
import torch
from kornia import create_meshgrid
import pdb
from tqdm import tqdm


def test_rays_dir_radii(ray_dirs):
    if type(ray_dirs) != torch.Tensor:
        ray_dirs = torch.Tensor(ray_dirs)
    dx_1 = torch.sqrt(torch.sum((ray_dirs[:-1, :, :] - ray_dirs[1:, :, :]) ** 2, -1))
    dx = torch.cat([dx_1, dx_1[-2:-1, :]], 0)
    radii = dx[..., None] * 2 / torch.sqrt(torch.tensor(12))

    return radii


def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        record_bytes,
        {
            "image_hash": tf.io.FixedLenFeature([], dtype=tf.int64),
            "cam_idx": tf.io.FixedLenFeature([], dtype=tf.int64),  # 0~12
            "equivalent_exposure": tf.io.FixedLenFeature([], dtype=tf.float32),
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "image": tf.io.FixedLenFeature([], dtype=tf.string),
            "ray_origins": tf.io.VarLenFeature(tf.float32),
            "ray_dirs": tf.io.VarLenFeature(tf.float32),
            "intrinsics": tf.io.VarLenFeature(tf.float32),
        }
    )


def get_cam_rays(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)  # (H, W, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions.numpy()


def get_rotate_one_image(cam_ray_dir, world_ray_dir):
    b_matrix = cam_ray_dir = cam_ray_dir.reshape(-1, 3)
    A_matrix = world_ray_dir = world_ray_dir.reshape(-1, 3)

    world_r123 = np.mat(world_ray_dir[:, :1]).reshape(-1, 1)
    world_r456 = np.mat(world_ray_dir[:, 1:2]).reshape(-1, 1)
    world_r789 = np.mat(world_ray_dir[:, 2:3]).reshape(-1, 1)

    cam_dir = np.mat(cam_ray_dir)
    r123 = np.linalg.lstsq(cam_dir, world_r123, rcond=None)[0]
    r456 = np.linalg.lstsq(cam_dir, world_r456, rcond=None)[0]
    r789 = np.linalg.lstsq(cam_dir, world_r789, rcond=None)[0]

    R = np.zeros([3, 3])
    R[0:1, :] = r123.T
    R[1:2, :] = r456.T
    R[2:3, :] = r789.T

    R_loss = world_ray_dir - cam_ray_dir @ R.T
    # print(f"Pose loss:\t{np.absolute(R_loss).mean()}")  # should < 0.01
    return R.tolist()


def handle_one_record(tfrecord, train_index, val_index):
    dataset = tf.data.TFRecordDataset(
        tfrecord,
        compression_type="GZIP",
    )
    dataset_map = dataset.map(decode_fn)

    train_or_val = 'train' in tfrecord
    
    if train_or_val:
        image_folder = train_img_folder
        meta_folder = train_meta_folder
        index = train_index
        train_index += 1
    else:
        image_folder = val_img_folder
        meta_folder = val_meta_folder
        index = val_index
        val_index += 1
    
    for batch in dataset_map:
        image_name = str(int(batch["image_hash"]))

        imagestr = batch["image"]
        image = tf.io.decode_png(imagestr, channels=0, dtype=tf.dtypes.uint8, name=None)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_folder, f"{image_name}.png"), image)

        cam_idx = int(batch["cam_idx"])
        equivalent_exposure = float(batch["equivalent_exposure"])
        height, width = int(batch["height"]), int(batch["width"])
        intrinsics = tf.sparse.to_dense(batch["intrinsics"]).numpy().tolist()
        ray_origins = tf.sparse.to_dense(batch["ray_origins"]).numpy().reshape(height, width, 3)
        ray_dirs = tf.sparse.to_dense(batch["ray_dirs"]).numpy().reshape(height, width, 3)

        K = np.zeros((3, 3), dtype=np.float32)
        # fx=focal,fy=focal,cx=img_w/2,cy=img_h/2
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = width * 0.5
        K[1, 2] = height * 0.5
        K[2, 2] = 1
        cam_ray_dir = get_cam_rays(height, width, K) # get normalized rays
        world_ray_dir = np.array(ray_dirs)
        rotate_m = get_rotate_one_image(cam_ray_dir, world_ray_dir)
        c2w_matrix = np.zeros([3, 4])
        c2w_matrix[:, :3] = rotate_m
        # average the origin as the final origin
        cur_origin = torch.mean(torch.mean(torch.tensor(ray_origins), dim=0), dim=0).tolist()
        c2w_matrix[:, 3:] = np.array(cur_origin).reshape(3, 1)

        meta_data_dict = {
            'W': width,
            'H': height,
            "intrinsics": torch.tensor(intrinsics),
            "c2w": torch.tensor(c2w_matrix),
            'image_name': image_name,
            "cam_idx": cam_idx,
            "equivalent_exposure": equivalent_exposure,
            "origin_pos": torch.tensor(ray_origins[0][0].tolist()),
            "index": index
        }
        if train_or_val:
            train_meta[image_name] = meta_data_dict
        else:
            val_meta[image_name] = meta_data_dict
        torch.save(train_meta, os.path.join(train_folder, "train_all_meta.pt"))
        torch.save(val_meta, os.path.join(val_folder, "val_all_meta.pt"))
        torch.save(meta_data_dict, os.path.join(meta_folder, image_name + ".pt"))
    return train_index, val_index

if __name__ == "__main__":
    waymo_root_p = "data/v1.0"
    result_root_folder = "data/pytorch_waymo_dataset"
    os.makedirs(result_root_folder, exist_ok=True)

    coordinate_info = {
        'origin_drb': [0.0, 0.0, 0.0],
        'pose_scale_factor': 1.0
    }
    torch.save(coordinate_info, os.path.join(result_root_folder, "coordinates.pt"))
    train_folder = os.path.join(result_root_folder, 'train')
    os.makedirs(train_folder, exist_ok=True)
    val_folder = os.path.join(result_root_folder, 'val')
    os.makedirs(val_folder, exist_ok=True)

    train_img_folder = os.path.join(train_folder, 'rgbs')
    train_meta_folder = os.path.join(train_folder, 'metadata')
    val_img_folder = os.path.join(val_folder, 'rgbs')
    val_meta_folder = os.path.join(val_folder, 'metadata')

    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(val_img_folder, exist_ok=True)
    os.makedirs(train_meta_folder, exist_ok=True)
    os.makedirs(val_meta_folder, exist_ok=True)

    train_meta, val_meta = {}, {}
    train_index = 0
    val_index = 0
    ori_waymo_data = sorted(glob.glob(os.path.join(waymo_root_p, "*")))
    for idx, tfrecord in enumerate(tqdm(ori_waymo_data)):
        train_index, val_index = handle_one_record(tfrecord, train_index, val_index)
