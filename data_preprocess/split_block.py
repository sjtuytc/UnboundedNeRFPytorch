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


def get_hparams():
    parser = argparse.ArgumentParser()

    parser.add_argument('--radius', type=float, default=0.3,
                        help='The radius of a block')
    parser.add_argument('--overlap', type=float,
                        default=0.5, help='overlap each block')
    parser.add_argument('--visualization', type=bool, default=True,
                        help="Whether visualize the split results")
    parser.add_argument('--visual_Block', type=bool, default=False,
                        help="When visualize whether visualize the split result")

    return vars(parser.parse_args())


def extract_imgname_origins(meta):
    img_origins = {}

    for idx, img_name in enumerate(meta):
        img_info = meta[img_name]
        origin = img_info['origin_pos']
        img_origins[img_name] = origin
    return img_origins


def resort_origins(img_origins, positions):
    # switch key and values in img_origins
    origin2name = {}
    for img_orin in img_origins:
        origin = img_origins[img_orin]
        origin2name[tuple(np.array(origin))] = img_orin

    sorted_origins = {}
    for pos in positions:
        sorted_origins[origin2name[tuple(np.array(pos))]] = pos

    return sorted_origins


def get_the_distance(r=2, overlap=0.5):  
    x = r * 0.9
    x0 = x
    # fd is the derivation of f
    f = 2 * np.arccos(x0 / r) * (r ** 2) - 2 * x0 * \
        np.sqrt(r ** 2 - x0 ** 2) - overlap * np.pi * r ** 2
    fd = (2 * x0 ** 2 - 2 * r ** 2) / np.sqrt(r **
                                              2 - x0 ** 2) - 2 * np.sqrt(r ** 2 - x0 ** 2)
    h = f / fd
    x = x0 - h
    # optimize the function using Newton Optimization
    while abs(x - x0) >= 1e-6:
        x0 = x
        f = 2 * np.arccos(x0 / r) * (r ** 2) - 2 * x0 * \
            np.sqrt(r ** 2 - x0 ** 2) - overlap * np.pi * r ** 2
        fd = (2 * x0 ** 2 - 2 * r ** 2) / np.sqrt(r **
                                                  2 - x0 ** 2) - 2 * np.sqrt(r ** 2 - x0 ** 2)
        h = f / fd
        x = x0 - h
    return 2 * x


def get_each_block_element_train(img_train_origins, centroid, radius):
    block_train_element = []

    index = 0
    for img_origin in img_train_origins:
        if np.linalg.norm(img_train_origins[centroid] - img_train_origins[img_origin]) <= radius:
            img_element = [img_origin, index]
            block_train_element.append(img_element)
            index += 1

    return block_train_element


def extract_img_base_camidx(cam_idx, train_meta):
    img_name = []
    for meta in train_meta:
        img_info = train_meta[meta]
        if img_info['cam_idx'] == cam_idx:
            img_name.append(meta)
    return img_name


def get_block_idx(img_name, split_train_results):
    for block in split_train_results:
        elements = split_train_results[block]["elements"]
        for element in elements:
            if element[0] == img_name:
                return [block, element[1]]
    return None


def get_val_block_index(img_val_origins, train_meta, val_meta, img_train_origins, split_train_results):
    split_val_results = defaultdict(list)
    for origin in img_val_origins:
        # find the corresponding camera first
        img_info = val_meta[origin]
        cam_idx = img_info['cam_idx']
        # fetch all the same index in train_meta
        img_list = extract_img_base_camidx(cam_idx, train_meta)
        # calculate the nearest img
        distance = 1000
        img_nearest = None
        for img in img_list:
            distance_temp = np.linalg.norm(img_val_origins[origin] - img_train_origins[img])
            if distance_temp < distance:
                distance = distance_temp
                img_nearest = img
        block_name, index = get_block_idx(img_nearest, split_train_results)
        '''
        block_info = {
            'elements': block_train_element,
            "centroid": [centroid, img_train_origins_resort[centroid].tolist()]
        }
        split_train_results[f'block_{idx}'] = block_info
        '''
        # fetch the nearest img_name and find its block and corresponding index
        split_val_results[block_name].append([origin, index])

    return split_val_results


def split_dataset(train_meta, val_meta, radius=0.5, overlap=0.5):
    img_train_origins = extract_imgname_origins(train_meta)
    img_val_origins = extract_imgname_origins(val_meta)

    train_positions = np.array([np.array(value) for value in img_train_origins.values()])
    val_positions = np.array([np.array(value) for value in img_val_origins.values()])

    train_indices = np.argsort(train_positions[:, 1])
    val_indices = np.argsort(val_positions[:, 1])

    train_positions = train_positions[train_indices, :]
    val_positions = val_positions[val_indices, :]

    img_train_origins_resort = resort_origins(img_train_origins, train_positions)
    img_val_origins_resort = resort_origins(img_val_origins, val_positions)

    distance = get_the_distance(r=radius, overlap=overlap)
    print(f"The block distance is: {distance}.")

    origin_1 = train_positions[0]
    centroids = []

    # find the first centroid
    temp_origin = {}
    for index, origin in enumerate(img_train_origins_resort):
        if np.linalg.norm(origin_1 - img_train_origins_resort[origin]) > radius:
            centroids.append(temp_origin)
            break
        temp_origin = origin

    # get a new centroid since the beggining
    temp_origin = {}
    judge = False
    for idx, origin in enumerate(img_train_origins_resort):
        if origin != centroids[-1] and judge == False: # have not reached the first centroid
            continue
        else:
            judge = True
        if np.linalg.norm(img_train_origins_resort[centroids[-1]] - img_train_origins_resort[origin]) > distance:
            centroids.append(temp_origin)
        temp_origin = origin

    split_train_results = {}
    for idx, centroid in enumerate(centroids):
        # find all points within the range
        block_train_element = get_each_block_element_train(img_train_origins_resort, centroid, radius)
        block_info = {
            'elements': block_train_element,
            "centroid": [centroid, img_train_origins_resort[centroid].tolist()]
        }
        split_train_results[f'block_{idx}'] = block_info
    # find the closest val_origin under the same camera settings
    split_val_results = get_val_block_index(img_val_origins_resort, train_meta, val_meta, img_train_origins_resort,
                                            split_train_results)

    return split_train_results, split_val_results


def extract_origins(meta):
    origins = []
    cam_index = defaultdict(int)
    for img_name in meta:
        img_info = meta[img_name]
        origins.append(img_info['origin_pos'])
        cam_index[img_info['cam_idx']] += 1

    print(cam_index)
    origins = np.array(origins)
    return origins


def extract_cam_idx(train_meta):
    cam_idx=[]
    for meta in train_meta:
        img_info=train_meta[meta]
        if img_info['cam_idx'] not in cam_idx:
            cam_idx.append(img_info['cam_idx'])
    return sorted(cam_idx)


def extract_img_each_idx(idx,train_meta,train_split_meta):
    imgs=[]
    for block in train_split_meta:
        for element in train_split_meta[block]['elements']:
            if train_meta[element[0]]['cam_idx']==idx:
                if element[0] not in imgs:
                    imgs.append(element[0])
    return imgs


def transfer_pt_to_json(pt_meta):
    new_dict = copy.deepcopy(pt_meta)
    for one_key in new_dict:
        new_dict[one_key]['intrinsics'] = np.array(new_dict[one_key]['intrinsics']).tolist()
        new_dict[one_key]['c2w'] = np.array(new_dict[one_key]['c2w']).tolist()
        new_dict[one_key]['origin_pos'] = np.array(new_dict[one_key]['origin_pos']).tolist()
    return new_dict

if __name__ == "__main__":
    args = get_hparams()
    print(args)
    root_dir = "data/pytorch_waymo_dataset"
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'val'), exist_ok=True)

    train_meta_path = os.path.join(root_dir, 'train', "split_block_train.json")
    val_meta_path = os.path.join(root_dir, 'val', "split_block_val.json")
    
    train_meta = torch.load(os.path.join(root_dir, 'train', 'train_all_meta.pt'))
    val_meta = torch.load(os.path.join(root_dir, 'val', 'val_all_meta.pt'))

    # rewrite to json following the convention
    with open(os.path.join(root_dir, 'train', 'train_all_meta.json'), "w") as fp:
        new_train_meta = transfer_pt_to_json(train_meta)
        json.dump(new_train_meta, fp)
        fp.close()
    with open(os.path.join(root_dir, 'val', 'val_all_meta.json'), "w") as fp:
        new_val_meta = transfer_pt_to_json(val_meta)
        json.dump(new_val_meta, fp)
        fp.close()

    print(
        f"Before spliting, there are {len(train_meta)} images for train and {len(val_meta)} images for val!")
    split_train_results, split_val_results = split_dataset(train_meta, val_meta, radius=args['radius'],
                                                           overlap=args['overlap'])
    print("Complete the split work!")

    block_train_json = {}
    block_val_json = {}

    for block in split_train_results:
        block_train_json[block] = split_train_results[block]
        print(f"{block} has {len(split_train_results[block]['elements'])}")
        with open(train_meta_path, "w") as fp:
            json.dump(block_train_json, fp)
            fp.close()

    for block in split_val_results:
        block_val_json[block] = split_val_results[block]
        print(f"{block} has {len(split_val_results[block])}")
        with open(val_meta_path, "w") as fp:
            json.dump(block_val_json, fp)
            fp.close()

    print(f"The split results has been stored in the {train_meta_path} and {val_meta_path}")

    if args['visualization']:
        train_origins = extract_origins(train_meta)
        val_origins = extract_origins(val_meta)
        # 11269,3
        # block_json
    
    cam_idxes=extract_cam_idx(train_meta)
    print(f"There are {len(cam_idxes)} cameras. ")
    cam_save_path = os.path.join(root_dir, "cam_info.json")
    cam_imgs={}
    for idx in cam_idxes:
        cam_imgs[idx]=extract_img_each_idx(idx, train_meta, split_train_results)
        with open(cam_save_path, "w") as fp:
            json.dump(cam_imgs, fp)
            fp.close()
    print(f"The camera information has been saved in the path of {cam_save_path}.")
    print("All done.")
