import os
import json
import gzip
import glob
import torch
import numpy as np
import imageio
import torch.nn.functional as F
import cv2


def load_co3d_data(cfg):

    # load meta
    with gzip.open(cfg.annot_path, 'rt', encoding='utf8') as zipfile:
        annot = [v for v in json.load(zipfile) if v['sequence_name'] == cfg.sequence_name]
    with open(cfg.split_path) as f:
        split = json.load(f)
        train_im_path = set()
        test_im_path = set()
        for k, lst in split.items():
            for v in lst:
                if v[0] == cfg.sequence_name:
                    if 'known' in k:
                        train_im_path.add(v[-1])
                    else:
                        test_im_path.add(v[-1])
    assert len(annot) == len(train_im_path) + len(test_im_path), 'Mismatch: '\
            f'{len(annot)} == {len(train_im_path) + len(test_im_path)}'

    # load datas
    imgs = []
    masks = []
    poses = []
    Ks = []
    i_split = [[], []]
    remove_empty_masks_cnt = [0, 0]
    for i, meta in enumerate(annot):
        im_fname = meta['image']['path']
        assert im_fname in train_im_path or im_fname in test_im_path
        sid = 0 if im_fname in train_im_path else 1
        if meta['mask']['mass'] == 0:
            remove_empty_masks_cnt[sid] += 1
            continue
        im_path = os.path.join(cfg.datadir, im_fname)
        mask_path = os.path.join(cfg.datadir, meta['mask']['path'])
        mask = imageio.imread(mask_path) / 255.
        if mask.max() < 0.5:
            remove_empty_masks_cnt[sid] += 1
            continue
        Rt = np.concatenate([meta['viewpoint']['R'], np.array(meta['viewpoint']['T'])[:,None]], 1)
        pose = np.linalg.inv(np.concatenate([Rt, [[0,0,0,1]]]))
        imgs.append(imageio.imread(im_path) / 255.)
        masks.append(mask)
        poses.append(pose)
        assert imgs[-1].shape[:2] == tuple(meta['image']['size'])
        half_image_size_wh = np.float32(meta['image']['size'][::-1]) * 0.5
        principal_point = np.float32(meta['viewpoint']['principal_point'])
        focal_length = np.float32(meta['viewpoint']['focal_length'])
        principal_point_px = -1.0 * (principal_point - 1.0) * half_image_size_wh
        focal_length_px = focal_length * half_image_size_wh
        Ks.append(np.array([
            [focal_length_px[0], 0, principal_point_px[0]],
            [0, focal_length_px[1], principal_point_px[1]],
            [0, 0, 1],
        ]))
        i_split[sid].append(len(imgs)-1)

    if sum(remove_empty_masks_cnt) > 0:
        print('load_co3d_data: removed %d train / %d test due to empty mask' % tuple(remove_empty_masks_cnt))
    print(f'load_co3d_data: num images {len(i_split[0])} train / {len(i_split[1])} test')

    imgs = np.array(imgs)
    masks = np.array(masks)
    poses = np.stack(poses, 0)
    Ks = np.stack(Ks, 0)
    render_poses = poses[i_split[-1]]
    i_split.append(i_split[-1])

    # visyalization hwf
    H, W = np.array([im.shape[:2] for im in imgs]).mean(0).astype(int)
    focal = Ks[:,[0,1],[0,1]].mean()

    return imgs, masks, poses, render_poses, [H, W, focal], Ks, i_split

