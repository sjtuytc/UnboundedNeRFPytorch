import os
import cv2
import pdb
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from block_nerf.rendering import *
from block_nerf.block_nerf_model import *
from block_nerf.block_nerf_lightning import *
from block_nerf.waymo_dataset import *
import imageio


def get_hparams():
    parser = ArgumentParser()
    parser.add_argument('--save_path', type=str,
                        default='data/result_pytorch_waymo',
                        help='result directory of dataset')

    parser.add_argument('--root_dir', type=str,
                        default='data/pytorch_waymo_dataset',
                        help='root directory of dataset')

    parser.add_argument('--ckpt_dir', type=str, default='data/ckpts',
                        help='path to load the trianed block checkpoints (e.g., block_1.ckpt).'
                        )

    parser.add_argument('--IDW_Power', type=int, default=1,
                        help='the value of the IDW power')

    parser.add_argument('--chunk', type=int, default=1024 * 2,
                        help='number of chunks')

    parser.add_argument('--cam_idx', type=list, default=[0],
                        help='the index of the camera you want to inference,0~11, total 12 cameras'
                        )

    return vars(parser.parse_args())


@torch.no_grad()
def batched_inference(
    model,
    embeddings,
    rays,
    ts,
    N_samples=128,
    N_importance=128,
    chunk=1024,
    use_disp=False,
    ):

    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        result_chunk = render_rays(
            model,
            embeddings,
            rays[i:i + chunk],
            ts[i:i + chunk],
            N_samples=N_samples,
            N_importance=N_importance,
            chunk=chunk,
            type='test',
            use_disp=use_disp,
            )
        for (k, v) in result_chunk.items():
            results[k] += [v.cpu()]
    for (k, v) in results.items():
        results[k] = torch.cat(v, 0)

    return results


def filter_cam_info_by_index(index, cam_infos):
    for (i, cam_info) in enumerate(cam_infos):
        if i == index:
            print('Now is inferencing the {cam_info} camera..')
            return cam_infos[cam_info]
    return None


def filter_Block(begin, blocks):
    block_filter = []
    for block in blocks:
        for element in blocks[block]['elements']:
            if element[0] == begin:
                block_filter.append(block)
    return block_filter


def DistanceWeight(point, centroid, p=4):
    point = point.numpy()
    centroid = np.array(centroid)
    return np.linalg.norm(point - centroid) ** -p


def Inverse_Interpolation(model_result, W_H):
    weights = []

    img_RGB = {}
    img_DEPTH = {}
    for block in model_result:
        block_RGB = np.clip(model_result[block]['rgb_fine'].view(H, W,
                            3).detach().numpy(), 0, 1)
        block_RGB = (block_RGB * 255).astype(np.uint8)
        img_RGB[block] = block_RGB

        block_depth = model_result[block]['depth_fine'].view(H,
                W).numpy()
        block_depth = np.nan_to_num(block_depth)  # change nan to 0
        mi = np.min(block_depth)  # get minimum depth
        ma = np.max(block_depth)
        block_depth = (block_depth - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        block_depth = (255 * block_depth).astype(np.uint8)
        img_DEPTH[block] = block_depth

        weights.append(model_result[block]['distance_weight'])

    weights = [weight / sum(weights) for weight in weights]
    print('The weight of each block is:', weights)
    img_pred = sum(weight * rgb for (weight, rgb) in zip(weights,
                   img_RGB.values())).astype(np.uint8)
    img_depth = sum(weight * depth for (weight, depth) in zip(weights,
                    img_DEPTH.values())).astype(np.uint8)

    img_RGB['compose'] = img_pred
    img_DEPTH['compose'] = img_depth

    return (img_RGB, img_DEPTH)


if __name__ == '__main__':
    print("Warning, this old implementation of BlockNeRF will be deprecated in the next version!")
    torch.cuda.empty_cache()
    hparams = get_hparams()
    os.makedirs(hparams['save_path'], exist_ok=True)

    block_split_info = None
    with open(os.path.join(hparams['root_dir'], 'train',
              'split_block_train.json'), 'r') as fp:
        block_split_info = json.load(fp)

    centroids = []
    for block in block_split_info:
        centroids.append(block_split_info[block]['centroid'])

    block_model = ['block_1', 'block_2']  # only render these models

    # block_model = ["block_6", "block_7"]

    with open(os.path.join(hparams['root_dir'], 'cam_info.json'), 'r'
              ) as fp:
        cam_infos = json.load(fp)

    (rgb_video_writer, depth_video_writer) = (None, None)

    for cam_idx in hparams['cam_idx']:
        print('Now is inferencing the {cam_idx} camera!')
        cam_infos = filter_cam_info_by_index(cam_idx, cam_infos)
        cam_info_begin = cam_infos[:-1]
        cam_info_end = cam_infos[1:]
        os.makedirs(os.path.join(hparams['save_path'], str(cam_idx)),
                    exist_ok=True)
        rgb_save_p = os.path.join(hparams['save_path'], str(cam_idx),
                                  'rgb_images')
        depth_save_p = os.path.join(hparams['save_path'], str(cam_idx),
                                    'depth_images')
        os.makedirs(rgb_save_p, exist_ok=True)
        os.makedirs(depth_save_p, exist_ok=True)

        # imgs = []
        # imgs_depth = []

        for i in tqdm(range(len(cam_info_begin))):
            (begin, end) = (cam_info_begin[i], cam_info_end[i])
            dataset = WaymoDataset(root_dir=hparams['root_dir'],
                                   split='compose', cam_begin=begin,
                                   cam_end=end)
            for j in tqdm(range(len(dataset))):
                batch = dataset[j]
                (rays, ts) = (batch['rays'], batch['ts'])
                (W, H) = batch['w_h']
                origin = rays[0, 0:3]
                blocks = filter_Block(begin, block_split_info)
                print('The current view belongs to the block of {blocks}.')
                model_result = {}
                for block in blocks:
                    if block in block_model:
                        ts[:] = \
                            find_idx_name(block_split_info[block]['elements'
                                ], begin)
                        print('Loading model ...')
                        model = \
                            Block_NeRF_System.load_from_checkpoint(os.path.join(hparams['ckpt_dir'
                                ], str(block) + '.ckpt')).cuda().eval()
                        models = {'block_model': model.Block_NeRF,
                                  'visibility_model': model.Visibility}
                        print("Model loaded. Now is inferring the {0}'s model.".format(block))
                        results = batched_inference(
                            models,
                            model.Embedding,
                            rays.cuda(),
                            ts.cuda(),
                            use_disp=model.hparams['use_disp'],
                            N_samples=model.hparams['N_samples'] * 2,
                            N_importance=model.hparams['N_importance']
                                * 2,
                            chunk=hparams['chunk'],
                            )
                        print("Finished inferring the {0}'s model.".format(block))
                        if results['transmittance_fine_vis'].mean() \
                            > 0.05:
                            results['distance_weight'] = \
                                DistanceWeight(point=origin,
                                    centroid=block_split_info[block]['centroid'
                                    ][1], p=hparams['IDW_Power'])
                            model_result[block] = results
                if not len(model_result):
                    continue
                (RGB_compose, Depth_compose) = \
                    Inverse_Interpolation(model_result, [W, H])
                if rgb_video_writer is None:
                    rgb_video_path = os.path.join(hparams['save_path'],
                            str(cam_idx), 'rgb_video.mp4')
                    depth_video_path = os.path.join(hparams['save_path'
                            ], str(cam_idx), 'depth_video.mp4')
                    (height, width) = RGB_compose['compose'].shape[:2]
                    rgb_video_writer = cv2.VideoWriter(rgb_video_path,
                            cv2.VideoWriter_fourcc(*'mp4v'), 15,
                            (width, height))
                    depth_video_writer = \
                        cv2.VideoWriter(depth_video_path,
                            cv2.VideoWriter_fourcc(*'mp4v'), 15,
                            (width, height))

                # imgs.append(RGB_compose['compose'])
                # imgs_depth.append(Depth_compose['compose'])

                rgb_video_writer.write(RGB_compose['compose'])
                depth_video_writer.write(Depth_compose['compose'])

                # save each rendered image

                for (RGB, Depth) in zip(RGB_compose, Depth_compose):
                    imageio.imwrite(os.path.join(rgb_save_p,
                                    '{0}_{1}_{2}_{3}.png'.format(i,
                                    begin, end, RGB)), RGB_compose[RGB])
                    imageio.imwrite(os.path.join(depth_save_p,
                                    '{0}_{1}_{2}_{3}_depth.png'.format(i,
                                    begin, end, Depth)),
                                    Depth_compose[Depth])
        if rgb_video_writer is not None:
            rgb_video_writer.release()
            depth_video_writer.release()
    print('All done.')
