## organize the images and ground-truths by camera positions and indexes
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


root_dir = "data/pytorch_waymo_dataset"

with open(os.path.join(root_dir, "train", f'train_all_meta.json'), 'r') as fp:
    train_all_meta = json.load(fp)

with open(os.path.join(root_dir, "val", f'val_all_meta.json'), 'r') as fp:
    val_all_meta = json.load(fp)

save_path = Path(f"data/sep19_ordered_dataset")
save_path.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.join(save_path, 'images_train'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'images_val'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'images_test'), exist_ok=True)


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
    
    
def form_unified_dict(old_to_new, metas, save_prefix='images_train', split_prefix='train'):
    file_paths = []
    c2ws = []
    widths = []
    heights = []
    focals = []
    nears = []
    fars = []
    return_metas = []
    positions = []
    intrisincs = []
    cam_idxs = []
    for idx, one_img in enumerate(tqdm(metas)):
        ori_path = os.path.join('data', 'pytorch_waymo_dataset', split_prefix, 'rgbs/' + one_img + ".png")
        new_name = old_to_new[one_img]
        if 'train' in split_prefix:
            cur_meta = train_all_meta[one_img]
        else:
            cur_meta = val_all_meta[one_img]
        cam_idxs.append(cur_meta['cam_idx'])
        final_path = os.path.join(save_prefix, new_name + ".png")
        full_save_path = os.path.join(save_path, final_path)
        if COPYFILE:
            shutil.copyfile(ori_path, full_save_path)
        file_paths.append(final_path)
        return_metas.append(cur_meta)
        positions.append(cur_meta['origin_pos'])
        if len(cur_meta['c2w']) < 4:
            cur_meta['c2w'].append([0.0, 0.0, 0.0, 1.0])
        c2ws.append(cur_meta['c2w'])
        widths.append(cur_meta['W'])
        heights.append(cur_meta['H'])
        focals.append(cur_meta['intrinsics'][0])
        nears.append(0.01)
        fars.append(15.)
        K = np.array([
            [cur_meta['intrinsics'][0], 0, 0.5*cur_meta['W']],
            [0, cur_meta['intrinsics'][0], 0.5*cur_meta['H']],
            [0, 0, 1]
        ]).tolist()
        intrisincs.append(K)

    lossmult = np.ones(np.array(heights).shape).tolist()
    pix2cam = get_pix2cam(focals=np.array(focals), width=np.array(widths), height=np.array(heights))
    positions = np.array(positions).tolist()
    return_metas = {'file_path': file_paths, 'cam2world': np.array(c2ws).tolist(), 'width': np.array(widths).tolist(),
    'height': np.array(heights).tolist(), 'focal': np.array(focals).tolist(), 'pix2cam': pix2cam, 'lossmult': lossmult, 
    'near':nears, 'far': fars, 'K': intrisincs, 'cam_idx': cam_idxs, 'position': positions}
    return return_metas


train_old_to_new = reorder_meta(train_all_meta)
val_old_to_new = reorder_meta(val_all_meta)

train_dict = form_unified_dict(train_old_to_new, train_all_meta, save_prefix='images_train', split_prefix='train')
val_dict = form_unified_dict(val_old_to_new, val_all_meta, save_prefix='images_val', split_prefix='val')
val_dict = form_unified_dict(val_old_to_new, val_all_meta, save_prefix='images_test', split_prefix='val')

# waymo does not have a test split
unified_meta = {'train': train_dict, 'val': val_dict, 'test': val_dict}

with open(os.path.join(save_path, 'metadata.json'), "w") as fp:
    json.dump(unified_meta, fp)
    fp.close()

print("All done!")

