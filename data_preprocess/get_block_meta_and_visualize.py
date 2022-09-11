import glob
import os
import json
from turtle import onkey
import numpy as np
import argparse
from collections import defaultdict
import open3d as o3d
import random
import copy
import pdb
import torch
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

train_imgs = train_block_split['block_' + str(block_index)]['elements']
val_imgs = val_block_split['block_' + str(block_index)]

file_paths = []
c2ws = []
widths = []
heights = []
focals = []

for idx, one_img in enumerate(train_imgs):
    file_paths.append('train/rgbs' + one_img[0] + ".png")
    cur_meta = train_all_meta[one_img[0]]
    cur_meta['c2w'].append([0.0, 0.0, 0.0, 1.0])
    c2ws.append(cur_meta['c2w'])
    widths.append(cur_meta['W'])
    heights.append(cur_meta['H'])
    focals.append(cur_meta['intrinsics'][0])
    
train_dict = {'file_path': file_paths, 'cam2world': np.array(c2ws).tolist(), 'width': np.array(widths).tolist(),
'height': np.array(heights).tolist(), 'focal': np.array(focals).tolist()}

file_paths = []
c2ws = []
widths = []
heights = []
focals = []

for idx, one_img in enumerate(val_imgs):
    file_paths.append('val/rgbs' + one_img[0] + ".png")
    cur_meta = val_all_meta[one_img[0]]
    cur_meta['c2w'].append([0.0, 0.0, 0.0, 1.0])
    c2ws.append(cur_meta['c2w'])
    widths.append(cur_meta['W'])
    heights.append(cur_meta['H'])
    focals.append(cur_meta['intrinsics'][0])
    
val_dict = {'file_path': file_paths, 'cam2world': np.array(c2ws).tolist(), 'width': np.array(widths).tolist(),
'height': np.array(heights).tolist(), 'focal': np.array(focals).tolist()}

# waymo does not have a test split
unified_meta = {'train': train_dict, 'val': val_dict, 'test': val_dict}

with open(os.path.join(save_path, 'metadata.json'), "w") as fp:
    json.dump(unified_meta, fp)
    fp.close()
pdb.set_trace()

