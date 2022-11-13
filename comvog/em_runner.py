# running the em procedure, jointly optimizing nerf and poses
import os
import pdb
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from comvog.bbox_compute import compute_bbox_by_cam_frustrm
from comvog.run_train import scene_rep_reconstruction
from comvog.run_render import run_render
from comvog.pose_utils.linemod_evaluator import LineMODEvaluator
from comvog.load_linemod import get_projected_points

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_ply_model(model_path, scale=1):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']*scale
    y = data['y']*scale
    z = data['z']*scale
    model = np.stack([x, y, z], axis=-1)
    return model


class NeRFEM(nn.Module):
    def __init__(self, args, cfg, data_dict):
        self.args = args
        self.args.no_reload = True
        self.cfg = cfg
        self.data_dict = data_dict
        self.sample_poses = []
        data_root = cfg.data.datadir
        seq_name = cfg.data.seq_name
        ply_path = os.path.join(data_root, 'models', seq_name, seq_name + ".ply")
        model = get_ply_model(ply_path)
        self.lm_evaluator = LineMODEvaluator(class_name=cfg.data.seq_name, obj_m=model)
        # self.lm_evaluator = LineMODEvaluator(class_name=cfg.data.seq_name, obj_m=data_dict['obj_m'])
        
    def set_poses(self, ):
        pdb.set_trace()
        
    def m_step(self, ):
        # train the nerfs
        args, cfg = self.args, self.cfg
        os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
        with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

        # coarse geometry searching (originally only for inward bounded scenes, extended to support waymo)
        # xyz_min_fine = torch.tensor([-1.0, -1.0, -1.0])
        # xyz_max_fine = torch.tensor([1.0, 1.0, 1.0])
        xyz_min_fine, xyz_max_fine = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **self.data_dict)
        if cfg.coarse_train.N_iters > 0:
            raise RuntimeError("Coarse train in EM is not supported!")

        # fine detail reconstruction for each sampled poses
        for idx, pose in enumerate(self.sample_poses):
            print(f"Running {idx}-th sample ...")
            self.data_dict['poses'] = pose
            psnr = scene_rep_reconstruction(
                    args=args, cfg=cfg,
                    cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
                    xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
                    data_dict=self.data_dict, stage='fine',
                    coarse_ckpt_path=None)
            run_render(args=args, cfg=cfg, data_dict=self.data_dict, device=device, add_info=str(psnr))
        return psnr
    
    def e_step(self, ):
        """
        Update the pose estimations via the results from m step.
        """
        poses, i_train = self.data_dict['poses'], self.data_dict['i_train']
        sample_poses, sample_rotations, sample_positions = [], [], [] 
        for i in range(self.cfg.nerf_em.sample_num):
            sample_poses.append(poses)
        self.sample_poses = sample_poses
    
    def run_em(self):
        gts, images = self.data_dict['gts'], self.data_dict['images']
        for idx, gt in enumerate(tqdm(gts)):
            index = gt['index']
            cur_pose_gt = gt['gt_pose']
            posecnn_results = gt['pose_noisy_rendered']
            ret = self.lm_evaluator.evaluate_linemod(cur_pose_gt, posecnn_results, gt['K'])
            # gt_vis = get_projected_points(cur_pose_gt, gt['K'], self.lm_evaluator.model, images[index].cpu().numpy(), post_str="gt")
            # posecnn_vis = get_projected_points(posecnn_results, gt['K'], self.lm_evaluator.model, images[index].cpu().numpy(), post_str="posecnn")
        self.lm_evaluator.summarize()    

        # # iteratively run e step and m step
        # self.e_step()
        # self.m_step()
        # print(f"Training id {i} ...")
        # poses = sample_poses()
        # data_dict, args = set_poses(args=args, cfg=cfg)
        # # M steps
        # psnr = run_train(args, cfg, data_dict, export_cam=True, export_geometry=True)
        # run_render(args=args, cfg=cfg, data_dict=data_dict, device=device, add_info=str(psnr))
        