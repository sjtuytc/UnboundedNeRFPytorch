# running the em procedure, jointly optimizing nerf and poses
import os
import pdb
import time
import torch
from math import ceil
import imageio
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from comvog.bbox_compute import compute_bbox_by_cam_frustrm
from comvog.run_train import scene_rep_reconstruction
from comvog.run_render import run_render
from comvog.pose_utils.linemod_evaluator import LineMODEvaluator
from comvog.pose_utils.pose_operators import pose_rot_interpolation, pose_sample
from comvog.load_linemod import get_projected_points
from comvog.pose_utils.visualization import *
from comvog.pose_utils.image_operators import *
from comvog import utils, dvgo, dcvgo, dmpigo
from comvog.run_render import render_viewpoints


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_ply_model(model_path, scale=1):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']*scale
    y = data['y']*scale
    z = data['z']*scale
    model = np.stack([x, y, z], axis=-1)
    return model


def get_canonical_t(pose):
    pose = pose.cpu().numpy()
    rot, trans = pose[:3, :3], pose[:3, -1]
    inv_rot = np.linalg.inv(rot)
    can_t = (inv_rot@trans).astype(np.int)
    return can_t


class NeRFEM(nn.Module):
    def __init__(self, args, cfg, data_dict):
        super(NeRFEM, self).__init__()
        self.args = args
        self.args.no_reload = True
        self.cfg = cfg
        self.data_dict = data_dict
        self.sample_poses = []
        data_root = cfg.data.datadir
        seq_name = cfg.data.seq_name
        ply_path = os.path.join(data_root, 'models', seq_name, seq_name + ".ply")
        model = get_ply_model(ply_path)
        # setup exp dir
        self.exp_dir = os.path.join(self.cfg.basedir, self.cfg.expname, self.cfg.pose_expname)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.lm_evaluator = LineMODEvaluator(class_name=cfg.data.seq_name, obj_m=model)
        # load pretrained NeRF model
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega" or cfg.data.dataset_type == "nerfpp":
            model_class = ComVoGModel
        elif cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        self.nerf_model = utils.load_model(model_class, ckpt_path).to(device)
        # Below are some testing functions
        poses, syn_images = data_dict['poses'], data_dict['syn_images']
        self.canonical_t = get_canonical_t(poses[0])  # [0, 0, 400]
        # poses = pose_rot_interpolation(poses[0], poses[15])
        # rgbs = self.render_many_views(poses)
        # imageio.imwrite("pose0.png", (syn_images[0]*255).cpu().numpy().astype(np.uint8))
        # imageio.imwrite("pose15.png", (syn_images[15]*255).cpu().numpy().astype(np.uint8))
        # rgb = self.render_a_view(poses[0])
        # imageio.imwrite("render0.png", (rgb*255).astype(np.uint8))
        
    def render_a_view(self, pose):
        HW = self.data_dict['HW']
        Ks = self.data_dict['Ks']
        render_viewpoints_kwargs = {
            'model': self.nerf_model,
            'ndc': self.cfg.data.ndc,
            'render_kwargs': {
                'near': self.data_dict['near'],
                'far': self.data_dict['far'],
                'bg': 1 if self.cfg.data.white_bkgd else 0,
                'stepsize': self.cfg.fine_model_and_render.stepsize,
                'inverse_y': self.cfg.data.inverse_y,
                'flip_x': self.cfg.data.flip_x,
                'flip_y': self.cfg.data.flip_y,
                'render_depth': True,
            },
        }
        rgbs, _, _ = render_viewpoints(cfg=self.cfg, render_poses=[pose], 
                                       HW=[HW[0]], Ks=[Ks[0]], gt_imgs=None, savedir=None, 
                                       dump_images=self.args.dump_images, **render_viewpoints_kwargs, verbose=False)
        return rgbs[0]
    
    def render_many_views(self, poses, video_name='test_render'):
        HW = self.data_dict['HW']
        Ks = self.data_dict['Ks']
        render_viewpoints_kwargs = {
            'model': self.nerf_model,
            'ndc': self.cfg.data.ndc,
            'render_kwargs': {
                'near': self.data_dict['near'],
                'far': self.data_dict['far'],
                'bg': 1 if self.cfg.data.white_bkgd else 0,
                'stepsize': self.cfg.fine_model_and_render.stepsize,
                'inverse_y': self.cfg.data.inverse_y,
                'flip_x': self.cfg.data.flip_x,
                'flip_y': self.cfg.data.flip_y,
                'render_depth': True,
            },
        }
        HWs = [HW[0] for pose in poses]
        Ks = [Ks[0] for pose in poses]
        rgbs, _, _ = render_viewpoints(cfg=self.cfg, render_poses=poses, 
                                       HW=HWs, Ks=Ks, gt_imgs=None, savedir=self.exp_dir, 
                                       dump_images=self.args.dump_images, **render_viewpoints_kwargs)
        save_folder = os.path.join(self.exp_dir, 'rendered_video')
        os.makedirs(save_folder, exist_ok=True)
        if video_name is not None:
            save_p = os.path.join(save_folder, video_name + ".mp4")
            imageio.mimwrite(save_p, utils.to8b(rgbs), fps=15, quality=8)
            print(f"Rendered views at {save_folder}.")
        return rgbs

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
    
    def collect_res_poses(self, gts, images):
        rot_deltas, trans_deltas = [], []
        print("Collecting residual poses ...")
        for idx, gt in enumerate(tqdm(gts)):
            index = gt['index']
            cur_pose_gt = gt['gt_pose']
            posecnn_results = gt['pose_noisy_rendered']
            rot_gt, trans_gt, rot_init, trans_init = cur_pose_gt[:3, :3], cur_pose_gt[:3, -1], posecnn_results[:3, :3], posecnn_results[:3, -1]
            rot_delta = rot_gt @ np.linalg.inv(rot_init)
            trans_delta = trans_gt - trans_init
            rot_deltas.append(rot_delta)
            trans_deltas.append(trans_delta)
        rot_deltas = np.array(rot_deltas)
        trans_deltas = np.array(trans_deltas)
        return rot_deltas, trans_deltas
    
    def apply_res_poses(self, center_pred, rot_deltas, trans_deltas):
        final_poses = np.array([center_pred for i in range(len(rot_deltas))])
        applied_rot = rot_deltas @ final_poses[:, :3, :3]
        applied_deltas = final_poses[:, :3, -1] + trans_deltas
        final_poses[:, :3, :3] = applied_rot
        final_poses[:, :3, -1] = applied_deltas
        return final_poses
    
    def obj_pose_to_cannonical(self, obj_pose):
        rot, trans = obj_pose[:3, :3], obj_pose[:3, -1]
        rot_inv = np.linalg.inv(rot)
        final_pose = obj_pose.copy()
        final_pose[:3, :3] = rot_inv
        final_pose[:3, -1] = rot_inv @ self.canonical_t
        return final_pose
    
    def observed_to_canonical(self, full_image, obj_pose, cam_k, ):
        observed_points = get_projected_points(obj_pose, cam_k, self.lm_evaluator.model, one_img=None)
        xmin, xmax = observed_points[:, 0].min(), observed_points[:, 0].max()
        ymin, ymax = observed_points[:, 1].min(), observed_points[:, 1].max()
        observed_crop = full_image[int(ymin):ceil(ymax), int(xmin):ceil(xmax)]
        # imageio.imwrite("debug_gt_cropped.png", (observed_crop).astype(np.uint8))
        return observed_crop
    
    def preprocess_render_observed(self, render_rgb, observed_rgb, idx, open_vis=True):
        # crop render rgb
        mask_image, xmin, xmax, ymin, ymax = get_bbox_from_img(render_rgb, color_thre=0.1)
        cropped_rendered = render_rgb[int(ymin):ceil(ymax), int(xmin):ceil(xmax)]
        if open_vis:
            imageio.imwrite(os.path.join(self.exp_dir, f"{str(idx)}_cropped_rendered.png"), (cropped_rendered*255).astype(np.uint8))
        cropped_mask = mask_image[int(ymin):ceil(ymax), int(xmin):ceil(xmax)]
        # resize observed and apply the mask
        height, width, _ = cropped_rendered.shape
        resized_observed = cv2.resize(observed_rgb, (width, height))
        cropped_observed = apply_mask_on_img(resized_observed, cropped_mask)
        if open_vis:
            imageio.imwrite(os.path.join(self.exp_dir, f"{str(idx)}_cropped_observed.png"), (cropped_observed).astype(np.uint8))
        return cropped_rendered, cropped_observed
        
    def render_and_observe_by_pose(self, obj_pose, full_observed, cam_k, idx):
        # get rendered via NeRF
        canno_pose = self.obj_pose_to_cannonical(obj_pose)
        render_rgb = self.render_a_view(canno_pose)
        # transfer the observed to canonical
        observed_rgb = self.observed_to_canonical(full_observed, obj_pose, cam_k)
        render_rgb, observed_rgb = self.preprocess_render_observed(render_rgb, observed_rgb, idx)
        return render_rgb, observed_rgb
    
    def render_observe_dist(self, render_rgb, observed_rgb):
        render_rgb = render_rgb / render_rgb.max()
        observed_rgb = observed_rgb / observed_rgb.max()
        mse_loss = torch.nn.functional.mse_loss(torch.tensor(render_rgb), torch.tensor(observed_rgb))
        return mse_loss.item()
    
    def render_and_observe_dist_of_one_pose(self, pose, full_image, cam_k, idx):
        render_rgb, observed_rgb = self.render_and_observe_by_pose(pose, full_image, cam_k, idx)
        dist = self.render_observe_dist(render_rgb, observed_rgb)
        return dist
    
    def render_and_observe_dist_of_poses(self, poses, full_image, cam_k, idx):
        all_dists = []
        for idx, pose in enumerate(tqdm(poses)):
            dist = self.render_and_observe_dist_of_one_pose(pose, full_image, cam_k, idx)
            all_dists.append(dist)
        min_pos = np.argmin(all_dists)
        return all_dists, poses[min_pos]
    
    def run_em(self, visualization=True):
        gts, images, obj_bb8 = self.data_dict['gts'], self.data_dict['images'], self.data_dict['obj_bb8']
        rot_deltas, trans_deltas = self.collect_res_poses(gts, images, )
        print("Saving results to ", self.exp_dir)
        for idx, gt in enumerate(tqdm(gts)):
            index = gt['index']
            cur_pose_gt = gt['gt_pose']
            posecnn_results = gt['pose_noisy_rendered']
            cur_image = images[index].cpu().numpy()
            posecnn_proposals = self.apply_res_poses(posecnn_results, rot_deltas, trans_deltas)
            all_dists, our_pose_result = self.render_and_observe_dist_of_poses(posecnn_proposals, cur_image, gt['K'], idx)
            # normal procedure, evaluating one item
            ret = self.lm_evaluator.evaluate_linemod(cur_pose_gt, our_pose_result, gt['K'])
            res_display = str('%.3f' % ret['ang_err_euler'] + ', %.3f' % ret['trans_err'] + ', %.3f' % ret['add_value'] + str(ret['add_final']))
            res_post = str('%.3f' % ret['add_value']) + str(ret['add_final'])
            # # batch forward to evaluate multiple hypothesis
            # ret = self.lm_evaluator.evaluate_proposals(cur_pose_gt, posecnn_proposals, gt['K'])
            # res_display = str('%.3f' % ret['ang_err_euler'] + ', %.3f' % ret['trans_err'] + ', %.3f' % ret['add_value'] + str(ret['add_final']))
            # res_post = str('%.3f' % ret['add_value']) + str(ret['add_final'])
            if visualization:
                visualize_pose_prediction(cur_pose_gt, posecnn_results, gt['K'], obj_bb8, cur_image, save_root=self.exp_dir, pre_str=str(index) + "_", post_str='posecnn_' + res_post)
                visualize_pose_prediction(cur_pose_gt, our_pose_result, gt['K'], obj_bb8, cur_image, save_root=self.exp_dir, pre_str=str(index) + "_", post_str='ours_' + res_post)
                # gt_vis = get_projected_points(cur_pose_gt, gt['K'], self.lm_evaluator.model, images[index].cpu().numpy(), save_root=self.exp_dir, pre_str=str(index) + "_", post_str="gt_" + res_post)
                # posecnn_vis = get_projected_points(posecnn_results, gt['K'], self.lm_evaluator.model, images[index].cpu().numpy(), save_root=self.exp_dir, pre_str=str(index) + "_", post_str="posecnn_" + res_post)
        self.lm_evaluator.summarize()
        if self.args.sample_num > 0:
            print("Warning, this results hold only for a sample num of:", self.args.sample_num)
        
        # # iteratively run e step and m step
        # self.e_step()
        # self.m_step()
        # print(f"Training id {i} ...")
        # poses = sample_poses()
        # data_dict, args = set_poses(args=args, cfg=cfg)
        # # M steps
        # psnr = run_train(args, cfg, data_dict, export_cam=True, export_geometry=True)
        # run_render(args=args, cfg=cfg, data_dict=data_dict, device=device, add_info=str(psnr))
        