# this code is from rnnpose
import pdb
import numpy as np
from comvog.pose_utils.linemod_constants import *


def chordal_distance(R1,R2):
    return np.sqrt(np.sum((R1-R2)*(R1-R2))) 


def rotation_angle(R1, R2):
    return 2*np.arcsin(chordal_distance(R1,R2)/np.sqrt(8) )


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


class LineMODEvaluator:
    def __init__(self, class_name, obj_m, icp_refine=False):
        self.class_name = class_name
        self.icp_refine = icp_refine
        # model_path = os.path.join(os.path.dirname(os.path.abspath(
        #     __file__)), '../../linemod/LM6d_converted/models', class_name, class_name + '.ply')
        # self.model = pvnet_data_utils.get_ply_model(model_path)
        self.model = obj_m
        self.diameter = diameters[class_name] / 100

        self.proj2d = []
        self.add = []
        self.adds = [] #force sym
        self.add2 = []
        self.add5 = []
        self.cmd5 = []

        self.icp_proj2d = []
        self.icp_add = []
        self.icp_cmd5 = []

        self.mask_ap = []
        # self.pose_preds=[]

        self.height = 480
        self.width = 640

        # model = inout.load_ply(model_path)
        model = obj_m
        # model['pts'] = model['pts'] * 1000
        self.icp_refiner = icp_utils.ICPRefiner(
            model, (self.width, self.height)) if icp_refine else None

    def projection_2d(self, pose_pred, pose_targets, K, icp=False, threshold=5):
        model_2d_pred = project(self.model, K, pose_pred)
        model_2d_targets = project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(
            model_2d_pred - model_2d_targets, axis=-1))
        if icp:
            self.icp_proj2d.append(proj_mean_diff < threshold)
        else:
            self.proj2d.append(proj_mean_diff < threshold)

    def projection_2d_sym(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = project(self.model, K, pose_pred)
        model_2d_targets = project(self.model, K, pose_targets)
        proj_mean_diff=np.mean(find_nearest_point_distance(model_2d_pred,model_2d_targets))

        self.proj_mean_diffs.append(proj_mean_diff)
        self.projection_2d_recorder.append(proj_mean_diff < threshold)

    def add2_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.02):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(
            self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            # idxs = find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(
                model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(
                model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add2.append(mean_dist < diameter)

    def add5_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.05):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(
            self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            # idxs = find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(
                model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(
                model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add5.append(mean_dist < diameter)

    
    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(
            self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            # idxs = find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(
                model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(
                model_pred - model_targets, axis=-1))
        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
        translation_distance = np.linalg.norm(
            pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        if icp:
            self.icp_cmd5.append(translation_distance <
                                 5 and angular_distance < 5)
        else:
            self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[
            0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

        R_refined, t_refined = self.icp_refiner.refine(
            depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(
            depth, R_refined, t_refined, K.copy(), no_depth=True)

        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))

        return pose_pred

    def icp_refine_(self, pose, anno, output):
        depth = read_depth(anno['depth_path']).astype(np.uint16)
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask = mask.astype(np.int32)
        pose = pose.astype(np.float32)

        poses = np.zeros([1, 7], dtype=np.float32)
        poses[0, :4] = mat2quat(pose[:, :3])
        poses[0, 4:] = pose[:, 3]

        poses_new = np.zeros([1, 7], dtype=np.float32)
        poses_icp = np.zeros([1, 7], dtype=np.float32)

        fx = 572.41140
        fy = 573.57043
        px = 325.26110
        py = 242.04899
        zfar = 6.0
        znear = 0.25
        factor = 1000.0
        error_threshold = 0.01

        rois = np.zeros([1, 6], dtype=np.float32)
        rois[:, :] = 1

        self.icp_refiner.solveICP(mask, depth,
                                  self.height, self.width,
                                  fx, fy, px, py,
                                  znear, zfar,
                                  factor,
                                  rois.shape[0], rois,
                                  poses, poses_new, poses_icp,
                                  error_threshold
                                  )

        pose_icp = np.zeros([3, 4], dtype=np.float32)
        pose_icp[:, :3] = quat2mat(poses_icp[0, :4])
        pose_icp[:, 3] = poses_icp[0, 4:]

        return pose_icp

    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        # adds = np.mean(self.adds)
        add2 = np.mean(self.add2)
        add5 = np.mean(self.add5)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)
        seq_len=len(self.add)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('ADD2 metric: {}'.format(add2))
        print('ADD5 metric: {}'.format(add5))
        # print('ADDS metric: {}'.format(adds))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('mask ap70: {}'.format(ap))
        print('seq_len: {}'.format(seq_len))
        # if cfg.test.icp:
        if self.icp_refine:
            print('2d projections metric after icp: {}'.format(
                np.mean(self.icp_proj2d)))
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
            print('5 cm 5 degree metric after icp: {}'.format(
                np.mean(self.icp_cmd5)))
        self.proj2d = []
        self.add = []
        self.add2 = []
        self.add5 = []
        # self.adds = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_proj2d = []    
        self.icp_add = []
        self.icp_cmd5 = []

        # #save pose predictions
        # if len(self.pose_preds)> 0:
        #     np.save(f"{self.class_name}_pose_preds.npy",self.pose_preds)
        # self.pose_preds=[]

        return {'proj2d': proj2d, 'add': add, 'add2': add2, 'add5': add5,'cmd5': cmd5, 'ap': ap, "seq_len": seq_len}
        
    def evaluate_linemod(self, pose_gt, pose_pred, cam_k): # sample_correspondence_pairs=False, direct_align=False, use_cnnpose=True):
        # len_src_f = example['stack_lengths'][0][0]
        # assert len( example['lifted_points']) == 1, "TODO: support bs>1"
        # lifted_points = example['lifted_points'][0].squeeze(0)
        # model_points = example['original_model_points'][:len_src_f]

        # K = example["K"].cpu().numpy().squeeze()
        # R_pred = preds_dict['Ti_pred'].G[:,0, :3,:3].squeeze().detach().cpu().numpy()
        # t_pred = preds_dict['Ti_pred'].G[:,0, :3,3:].squeeze(0).detach().cpu().numpy()
        # pose_pred= preds_dict['Ti_pred'].G[:,0, :3].squeeze().detach().cpu().numpy()
        # pose_gt = example['original_RT'].squeeze()[:3].cpu().numpy()
        ang_err = rotation_angle(pose_gt[:3, :3], pose_pred[:3, :3])
        trans_err = np.linalg.norm(pose_pred[:3, -1:] - pose_gt[:3, -1:])  # 3x1
        if self.class_name in ['eggbox', 'glue']:
            self.add_metric(pose_pred, pose_gt, syn=True)
            self.add2_metric(pose_pred, pose_gt, syn=True)
            self.add5_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)
            self.add2_metric(pose_pred, pose_gt)
            self.add5_metric(pose_pred, pose_gt)

        self.projection_2d(pose_pred, pose_gt, K=cam_k)
        self.cm_degree_5_metric(pose_pred, pose_gt)

        # vis
        # pc_proj_vis = vis_pointclouds_cv2((pose_gt[:3, :3]@model_points.cpu().numpy(
        # ).T+pose_gt[:3, -1:]).T, example["K"].cpu().numpy().squeeze(), [480,640])
        # pc_proj_vis_pred = vis_pointclouds_cv2((pose_pred[:3, :3]@model_points.cpu().numpy(
        # ).T+pose_pred[:3, -1:]).T, example["K"].cpu().numpy().squeeze(), [ 480, 640])

        return {
            "ang_err": ang_err,
            "trans_err": trans_err,
            "pnp_inliers": -1,#len(inliers),
            # "pc_proj_vis": pc_proj_vis,
            # "pc_proj_vis_pred": pc_proj_vis_pred,
            # "keypoints_2d_vis": np.zeros_like(pc_proj_vis_pred) #keypoints_2d_vis
        }
