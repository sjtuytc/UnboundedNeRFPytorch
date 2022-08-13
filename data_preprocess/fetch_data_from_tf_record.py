import numpy as np
import datetime, os
from base64 import decodestring
import cv2
import os
import json
import glob
import tensorflow as tf
import torch


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


def handle_one_record(tfrecord, exist_imgs, index):
    dataset = tf.data.TFRecordDataset(
        tfrecord,
        compression_type="GZIP",
    )
    dataset_map = dataset.map(decode_fn)

    os.makedirs(result_root_folder, exist_ok=True)
    meta_folder = os.path.join(result_root_folder, 'json')
    image_folder = os.path.join(result_root_folder, "images")
    json_path = os.path.join(meta_folder, "train.json")
    os.makedirs(meta_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    for idx, batch in enumerate(dataset_map):
        print(f"\tLoading the {idx + 1}th image...")
        image_name = str(int(batch["image_hash"]))

        if image_name + ".png" in exist_imgs:
            print(f"\t{image_name}.png has been loaded!")
            continue

        index += 1
        imagestr = batch["image"]
        image = tf.io.decode_png(imagestr, channels=0, dtype=tf.dtypes.uint8, name=None)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_folder, f"{image_name}.png"), image)

        cam_idx = int(batch["cam_idx"])
        equivalent_exposure = float(batch["equivalent_exposure"])
        height, width = int(batch["height"]), int(batch["width"])
        intrinsics = tf.sparse.to_dense(batch["intrinsics"]).numpy()

        ray_origins = tf.sparse.to_dense(batch["ray_origins"]).numpy().reshape(height, width, 3)
        ray_dirs = tf.sparse.to_dense(batch["ray_dirs"]).numpy().reshape(height, width, 3)
        with open(os.path.join(image_folder, f"{image_name}_ray_origins.npy"), "wb") as f:
            np.save(f, ray_origins)
        with open(os.path.join(image_folder, f"{image_name}_ray_dirs.npy"), "wb") as f:
            np.save(f, ray_dirs)

        cur_data_dict = {
            "image_name": image_name + ".png",
            "cam_idx": cam_idx,
            "equivalent_exposure": equivalent_exposure,
            "height": height,
            "width": width,
            "intrinsics": intrinsics.tolist(),
            "origin_pos": ray_origins[0][0].tolist(),
            "index": index
        }

        train_meta[image_name] = cur_data_dict
        with open(json_path, "w") as fp:
            json.dump(train_meta, fp)
            fp.close()

    return index

def get_the_current_index(root_dir):
    if os.path.exists(os.path.join(root_dir, 'json/train.json')):
        with open(os.path.join(root_dir, 'json/train.json'), 'r') as fp:
            meta = json.load(fp)
        return len(meta)
    return 0


if __name__ == "__main__":
    waymo_root_p = "data/v1.0"
    result_root_folder = "data/pytorch_block_nerf_dataset"
    ori_waymo_data = sorted(glob.glob(os.path.join(waymo_root_p, "*")))
    exist_img_list = sorted(glob.glob(os.path.join(result_root_folder + "/images", "*.png")))

    exist_imgs = []
    for img_name in exist_img_list:
        exist_imgs.append(os.path.basename(img_name))

    index = get_the_current_index(result_root_folder)
    print(f"Has loaded {index} images!")
    train_meta = {}

    for idx, tfrecord in enumerate(ori_waymo_data):
        print(f"Handling the {idx+1}/{len(ori_waymo_data)} tfrecord")
        index = handle_one_record(tfrecord, exist_imgs, index)
