import os
import json
import numpy as np
import random
import copy
import pdb
import shutil
import torch
from tqdm import tqdm
from pathlib import Path

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


def form_unified_dict(images, metas, save_prefix='images_train', split_prefix='train'):
    file_paths = []
    c2ws = []
    widths = []
    heights = []
    focals = []
    nears = []
    fars = []
    metas = []
    positions = []
    for idx, one_img in enumerate(images):
        ori_path = os.path.join('data', 'pytorch_waymo_dataset', split_prefix, 'rgbs/' + one_img[0] + ".png")
        if 'train' in split_prefix:
            cur_meta = train_all_meta[one_img[0]]
        else:
            cur_meta = val_all_meta[one_img[0]]
        final_path = os.path.join(save_prefix, str(cur_meta['cam_idx']) + "_" + str(idx) + ".png")
        full_save_path = os.path.join(save_path, final_path)
        shutil.copyfile(ori_path, full_save_path)
        file_paths.append(final_path)
        metas.append(cur_meta)
        positions.append(cur_meta['origin_pos'])
        cur_meta['c2w'].append([0.0, 0.0, 0.0, 1.0])
        c2ws.append(cur_meta['c2w'])
        widths.append(cur_meta['W'])
        heights.append(cur_meta['H'])
        focals.append(cur_meta['intrinsics'][0])
        nears.append(0.01)
        fars.append(15.)
    lossmult = np.ones(np.array(heights).shape).tolist()
    pix2cam = get_pix2cam(focals=np.array(focals), width=np.array(widths), height=np.array(heights))
    positions = np.array(positions)
    metas = {'file_path': file_paths, 'cam2world': np.array(c2ws).tolist(), 'width': np.array(widths).tolist(),
    'height': np.array(heights).tolist(), 'focal': np.array(focals).tolist(), 'pix2cam': pix2cam, 'lossmult': lossmult, 
    'near':nears, 'far': fars}
    return metas


train_dict = form_unified_dict(train_imgs, train_all_meta, save_prefix='images_train', split_prefix='train')
val_dict = form_unified_dict(val_imgs, val_all_meta, save_prefix='images_val', split_prefix='val')
val_dict = form_unified_dict(val_imgs, val_all_meta, save_prefix='images_test', split_prefix='val')

# waymo does not have a test split
unified_meta = {'train': train_dict, 'val': val_dict, 'test': val_dict}

with open(os.path.join(save_path, 'metadata.json'), "w") as fp:
    json.dump(unified_meta, fp)
    fp.close()

print("All done!")
