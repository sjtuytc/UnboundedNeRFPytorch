# transfer formats from mega-nerf to YONO supported formats
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


COPYFILE = True  # change it to true to rename and copy image files


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


def reorder_meta(meta_dict):
    cam2images_pos = {}
    old_name_2_new_name = {}
    # collect images by cams
    for one_key in tqdm(meta_dict):
        cur_value = meta_dict[one_key]
        cam_idx = cur_value['cam_idx']
        if cam_idx not in cam2images_pos:
            cam2images_pos[cam_idx] = [[one_key, cur_value['origin_pos']]]
        else:
            cam2images_pos[cam_idx].append([one_key, cur_value['origin_pos']])
    for one_cam in cam2images_pos:
        pos_list = cam2images_pos[one_cam]
        pos_list.sort(key=lambda row: (row[1][1], row[1][0]))
        for idx, ele in enumerate(pos_list):
            old_name = ele[0]
            new_name = str(one_cam) + "_" + str(idx)
            old_name_2_new_name[old_name] = new_name
    return old_name_2_new_name
    
    
def form_unified_dict(meta_root, save_prefix='images_train', split_prefix='train'):
    all_metas = sorted(os.listdir(meta_root))
    file_paths = []
    c2ws = []
    widths = []
    heights = []
    distortion = []
    intrisincs = []
    rgb_root = train_rgb_p if 'train' in split_prefix else val_rgb_p
    meta_root = train_meta_p if 'train' in split_prefix else val_meta_p
    for idx, meta_p in enumerate(tqdm(all_metas)):
        if meta_p == f'{split_prefix}.pt':
            continue
        ori_path = os.path.join(rgb_root, meta_p.replace('.pt', ".jpg"))
        cur_meta = torch.load(os.path.join(meta_root, meta_p))
        cur_meta['c2w'] = cur_meta['c2w'].numpy().tolist()
        final_path = os.path.join(save_prefix, meta_p.replace('.pt', ".jpg"))
        full_save_path = os.path.join(save_dataset_root, final_path)
        if COPYFILE:
            shutil.copyfile(ori_path, full_save_path)
        file_paths.append(final_path)
        if len(cur_meta['c2w']) < 4:
            cur_meta['c2w'].append([0.0, 0.0, 0.0, 1.0])
            c2ws.append(cur_meta['c2w'])
        widths.append(cur_meta['W'])
        heights.append(cur_meta['H'])
        distortion.append(cur_meta['distortion'].numpy().tolist())
        intrisincs.append(cur_meta['intrinsics'].numpy().tolist())
    return_metas = {'file_path': file_paths, 'cam2world': np.array(c2ws).tolist(), 'width': np.array(widths).tolist(),
                    'height': np.array(heights).tolist(), 'K': intrisincs,}
    return return_metas


data_name = 'building'
root_dir = f"data/mega/{data_name}/{data_name}-pixsfm"
train_p = os.path.join(root_dir, 'train')
val_p = os.path.join(root_dir, 'val')
train_meta_p = os.path.join(train_p, 'metadata')
train_rgb_p = os.path.join(train_p, 'rgbs')
val_meta_p = os.path.join(val_p, 'metadata')
val_rgb_p = os.path.join(val_p, 'rgbs')

save_dataset_root = Path(os.path.join(f"data/oct9_meta", data_name))
save_dataset_root.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.join(save_dataset_root, 'images_train'), exist_ok=True)
os.makedirs(os.path.join(save_dataset_root, 'images_val'), exist_ok=True)
os.makedirs(os.path.join(save_dataset_root, 'images_test'), exist_ok=True)

train_dict = form_unified_dict(meta_root=train_meta_p, save_prefix='images_train', split_prefix='train')
val_dict = form_unified_dict(meta_root=val_meta_p, save_prefix='images_val', split_prefix='val')
val_dict = form_unified_dict(meta_root=val_meta_p, save_prefix='images_test', split_prefix='val')

# many mega datasets does not have a test split
unified_meta = {'train': train_dict, 'val': val_dict, 'test': val_dict}

with open(os.path.join(save_dataset_root, 'metadata.json'), "w") as fp:
    json.dump(unified_meta, fp)
    fp.close()

print("All done!")

