import numpy as np
import datetime, os
from base64 import decodestring
import cv2
import os
import matplotlib.pyplot as plt
import json
import torch
import glob
import pdb
import tensorflow as tf


def handle_one_record(record_p):
    train_or_val = "train" if "train" in record_p else "val"
    dataset = tf.data.TFRecordDataset(
        record_p,
        compression_type="GZIP",
    )
    dataset_map = dataset.map(decode_fn)
    
    os.makedirs(result_root_folder, exist_ok=True)
    meta_folder = os.path.join(result_root_folder, "gt")
    image_folder = os.path.join(result_root_folder, "images")
    cur_meta_file = os.path.join(meta_folder, train_or_val + ".json")
    os.makedirs(meta_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    for idx, batch in enumerate(dataset_map):
        print("Iterating:", idx, "...", end="\r")
        imagestr = batch["image"]
        image_hash = batch["image_hash"]
        image = tf.io.decode_png(imagestr, channels=0, dtype=tf.dtypes.uint8, name=None)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hash = str(int(image_hash))
        cam_idx = int(batch["cam_idx"])
        equi_exp = float(batch["equivalent_exposure"])
        height, width = int(batch["height"]), int(batch["width"])
        ray_origins = tf.sparse.to_dense(batch["ray_origins"]).numpy()
        ray_dirs = (
            tf.sparse.to_dense(batch["ray_dirs"]).numpy().reshape(height, width, 3)
        )
        intrinsics = tf.sparse.to_dense(batch["intrinsics"]).numpy()
        cur_data_dict = {
            "image_hash": image_hash,
            "cam_idx": cam_idx,
            "equivalent_exposure": equi_exp,
            "height": height,
            "width": width,
            "intrinsics": intrinsics.tolist(),
        }
        if image_hash in train_meta or image_hash in val_meta:
            raise RuntimeError("key collision, this should not happen!")
        if train_or_val == "train":
            train_meta[image_hash] = cur_data_dict
            with open(cur_meta_file, "w") as fp:
                json.dump(train_meta, fp)
                fp.close()
        else:
            val_meta[image_hash] = cur_data_dict
            with open(cur_meta_file, "w") as fp:
                json.dump(val_meta, fp)
                fp.close()
        # save image and meta data
        cv2.imwrite(os.path.join(image_folder, image_hash + "_rgb.png"), image)
        with open(
            os.path.join(image_folder, image_hash + "_ray_origins.npy"), "wb"
        ) as f:
            np.save(f, ray_origins)
        with open(os.path.join(image_folder, image_hash + "_ray_dirs.npy"), "wb") as f:
            np.save(f, ray_dirs)


# Read the data back out.
def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        {
            "image_hash": tf.io.FixedLenFeature([], dtype=tf.int64),
            "cam_idx": tf.io.FixedLenFeature([], dtype=tf.int64),
            "equivalent_exposure": tf.io.FixedLenFeature([], dtype=tf.float32),
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "image": tf.io.FixedLenFeature([], dtype=tf.string),
            "ray_origins": tf.io.VarLenFeature(tf.float32),
            "ray_dirs": tf.io.VarLenFeature(tf.float32),
            "intrinsics": tf.io.VarLenFeature(tf.float32),
        },
    )


if __name__ == '__main__':
    waymo_root_p = "data/v1.0"
    result_root_folder = "data/pytorch_block_nerf_dataset"
    ori_waymo_data = sorted(glob.glob(os.path.join(waymo_root_p, "*")))
    train_meta, val_meta = {}, {}

    for idx, record_p in enumerate(ori_waymo_data):
        print("Handling", idx, "/", len(ori_waymo_data), ".")
        handle_one_record(record_p)