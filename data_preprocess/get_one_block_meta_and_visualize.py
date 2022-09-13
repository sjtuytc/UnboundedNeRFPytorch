import glob
import os
import json
from turtle import onkey
from webbrowser import get
import numpy as np
import argparse
from collections import defaultdict
import open3d as o3d
import random
import copy
import pdb
import shutil
import torch
from tqdm import tqdm
from pathlib import Path

root_dir = "data/pytorch_waymo_dataset"
block_index = 0

with open(os.path.join(root_dir, "train", f'split_block_train.json'), 'r') as fp:
    train_block_split = json.load(fp)

with open(os.path.join(root_dir, "train", f'train_all_meta.json'), 'r') as fp:
    train_all_meta = json.load(fp)

with open(os.path.join(root_dir, "val", f'split_block_val.json'), 'r') as fp:
    val_block_split = json.load(fp)

with open(os.path.join(root_dir, "val", f'val_all_meta.json'), 'r') as fp:
    val_all_meta = json.load(fp)

save_path = Path(f"data/samples/block_{block_index}")
save_path.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.join(save_path, 'images_train'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'images_val'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'images_test'), exist_ok=True)

train_imgs = train_block_split['block_' + str(block_index)]['elements']
val_imgs = val_block_split['block_' + str(block_index)]

file_paths = []
c2ws = []
widths = []
heights = []
focals = []
nears = []
fars = []

for idx, one_img in enumerate(train_imgs):
    final_path = os.path.join('images_train', one_img[0] + ".png")
    file_paths.append(final_path)
    cur_meta = train_all_meta[one_img[0]]
    cur_meta['c2w'].append([0.0, 0.0, 0.0, 1.0])
    c2ws.append(cur_meta['c2w'])
    widths.append(cur_meta['W'])
    heights.append(cur_meta['H'])
    focals.append(cur_meta['intrinsics'][0])
    nears.append(0.01)
    fars.append(15.)

def get_pix2cam(focals, width, height):
    fx = np.array(focals)
    fy = np.array(focals)
    cx = np.array(width) * .5
    cy = np.array(height) * .5
    arr0 = np.zeros_like(cx)
    arr1 = np.ones_like(cx)
    k_inv = np.array([
        [arr1 / fx, arr0, -cx / fx],
        [arr0, -arr1 / fy, cy / fy],
        [arr0, arr0, -arr1],
    ])
    k_inv = np.moveaxis(k_inv, -1, 0)
    return k_inv.tolist()

lossmult = np.ones(np.array(heights).shape).tolist()
pix2cam = get_pix2cam(focals=np.array(focals), width=np.array(widths), height=np.array(heights))
train_dict = {'file_path': file_paths, 'cam2world': np.array(c2ws).tolist(), 'width': np.array(widths).tolist(),
'height': np.array(heights).tolist(), 'focal': np.array(focals).tolist(), 'pix2cam': pix2cam, 'lossmult': lossmult, 
'near':nears, 'far': fars}

file_paths = []
c2ws = []
widths = []
heights = []
focals = []

for idx, one_img in enumerate(val_imgs):
    final_path = os.path.join('images_val', one_img[0] + ".png")
    file_paths.append(final_path)
    cur_meta = val_all_meta[one_img[0]]
    cur_meta['c2w'].append([0.0, 0.0, 0.0, 1.0])
    c2ws.append(cur_meta['c2w'])
    widths.append(cur_meta['W'])
    heights.append(cur_meta['H'])
    focals.append(cur_meta['intrinsics'][0])

lossmult = np.ones(np.array(heights).shape).tolist()
pix2cam = get_pix2cam(focals=np.array(focals), width=np.array(widths), height=np.array(heights))
val_dict = {'file_path': file_paths, 'cam2world': np.array(c2ws).tolist(), 'width': np.array(widths).tolist(),
'height': np.array(heights).tolist(), 'focal': np.array(focals).tolist(), 'pix2cam': pix2cam, 'lossmult': lossmult,
'near':nears, 'far': fars}

# waymo does not have a test split
unified_meta = {'train': train_dict, 'val': val_dict, 'test': val_dict}

with open(os.path.join(save_path, 'metadata.json'), "w") as fp:
    json.dump(unified_meta, fp)
    fp.close()

print("Copying files to save path: ", str(save_path))
for one_img in tqdm(train_imgs):
    ori_path = os.path.join('data', 'pytorch_waymo_dataset', 'train/rgbs/' + one_img[0] + ".png")
    final_path = os.path.join(save_path, 'images_train', one_img[0] + ".png")
    shutil.copyfile(ori_path, final_path)

for one_img in tqdm(val_imgs):
    ori_path = os.path.join('data', 'pytorch_waymo_dataset', 'val/rgbs/' + one_img[0] + ".png")
    final_path = os.path.join(save_path, 'images_val', one_img[0] + ".png")
    shutil.copyfile(ori_path, final_path)
    final_path = os.path.join(save_path, 'images_test', one_img[0] + ".png")  # waymo does not have the test split
    shutil.copyfile(ori_path, final_path)

print("All done!")
