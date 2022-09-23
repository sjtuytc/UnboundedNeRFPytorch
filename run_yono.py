import os, sys, copy, glob, json, time, random, argparse
import mmcv
import numpy as np
import pdb
import torch
from yono.load_everything import load_everything
from yono.run_export_bbox import *
from yono.run_export_coarse import run_export_coarse
from yono.run_train import run_train
from yono.run_render import run_render
from yono.gen_cam_paths import gen_cam_paths


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--program', required=True, type=str, 
                        help='choose one program to run', choices=['export_bbox', 'export_coarse', 'render', 'train', 'gen_trace']
                        )
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--sample_num", type=int, default=-1,
                        help='Sample number of data points in the dataset, used for debugging.')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # render and eval options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


if __name__=='__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)
    program = args.program

    # launch the corresponding program
    if program == "export_bbox":
        run_export_bbox_cams(args=args, cfg=cfg, data_dict=data_dict)
    elif program == "export_coarse":
        run_export_coarse(args=args, cfg=cfg, device=device)
    elif program == "train":
        run_train(args, cfg, data_dict)
        # render after training
        run_render(args=args, cfg=cfg, data_dict=data_dict, device=device)
    elif program == 'render':
        run_render(args=args, cfg=cfg, data_dict=data_dict, device=device)
    elif program == 'gen_trace':
        gen_cam_paths(args=args, cfg=cfg, data_dict=data_dict)
    else:
        raise NotImplementedError(f"Program {program} is not supported!")
    print(f"Finished running program {program}.")
