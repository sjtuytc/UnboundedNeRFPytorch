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
from PIL import Image


COPYFILE = True  # change it to true to rename and copy image files
IMAGE_DOWNSCALE = 4
    
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
        img = Image.open(full_save_path).convert('RGB')
        if IMAGE_DOWNSCALE != 1:
            img = img.resize((cur_meta['W'] // IMAGE_DOWNSCALE, cur_meta['H'] // IMAGE_DOWNSCALE), 
                             Image.Resampling.LANCZOS)
        img = img.save(full_save_path)
        file_paths.append(final_path)
        if len(cur_meta['c2w']) < 4:
            cur_meta['c2w'].append([0.0, 0.0, 0.0, 1.0])
            c2ws.append(cur_meta['c2w'])
        widths.append(cur_meta['W'] / IMAGE_DOWNSCALE)
        heights.append(cur_meta['H'] / IMAGE_DOWNSCALE)
        
        distortion.append(cur_meta['distortion'].numpy().tolist())
        fx, fy, half_w, half_h = cur_meta['intrinsics'].numpy().tolist()
        fx, fy, half_w, half_h = fx / IMAGE_DOWNSCALE, fy / IMAGE_DOWNSCALE, half_w / IMAGE_DOWNSCALE, half_h / IMAGE_DOWNSCALE
        K = [[fx, 0, half_w, 0], [0, fy, half_h, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        intrisincs.append(K)
    return_metas = {'file_path': file_paths, 'cam2world': np.array(c2ws).tolist(), 'width': np.array(widths).tolist(),
                    'height': np.array(heights).tolist(), 'K': intrisincs,}
    return return_metas


# data_name = 'building'
# data_name = 'rubble'
data_name = 'quad'
root_dir = f"data/mega/{data_name}/{data_name}-pixsfm"
train_p = os.path.join(root_dir, 'train')
val_p = os.path.join(root_dir, 'val')
train_meta_p = os.path.join(train_p, 'metadata')
train_rgb_p = os.path.join(train_p, 'rgbs')
val_meta_p = os.path.join(val_p, 'metadata')
val_rgb_p = os.path.join(val_p, 'rgbs')

save_dataset_root = Path(os.path.join(f"data/oct9_mega", data_name))
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

