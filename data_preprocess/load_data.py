import tensorflow as tf
import numpy as np
import datetime, os
from base64 import decodestring
import cv2
import os
import matplotlib.pyplot as plt
import json
import torch
demo_data_p = "/content/drive/MyDrive/blocknerf/v1.0_waymo_block_nerf_mission_bay_train.tfrecord-00002-of-01063"
train_meta, val_meta = {}, {}
train_or_val = "train" if "train" in demo_data_p else "val"
filenames = [demo_data_p]
dataset = tf.data.TFRecordDataset(
    filenames[0],
    compression_type="GZIP",
)
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

dataset_map = dataset.map(decode_fn)