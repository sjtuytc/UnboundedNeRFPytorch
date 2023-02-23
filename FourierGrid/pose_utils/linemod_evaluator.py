# this code is from rnnpose
import pdb
import numpy as np
from FourierGrid.pose_utils.linemod_constants import *
from FourierGrid.pose_utils.pose_operators import *
from scipy.spatial.transform import Rotation as R

    
def rotation_angle_euler(R1, R2):
    if len(R2.shape)==2:
        # http://www.boris-belousov.net/2016/12/01/quat-dist/#:~:text=The%20difference%20rotation%20matrix%20that,matrix%20R%20%3D%20P%20Q%20%E2%88%97%20.
        rotation_difference = R1 @ np.linalg.inv(R2)
        theta = np.arccos((np.trace(rotation_difference) - 1) / 2)
        theta = np.rad2deg(theta)
        euler = R.from_matrix(rotation_difference).as_euler('zyx', degrees=True)
        norm_angle = np.linalg.norm(euler)
        return norm_angle
    else:  # batch mode
        rotation_difference = R1 @ np.linalg.inv(R2)
        batch_norm_angles = [rot_diff_to_norm_angle(rot_d) for rot_d in rotation_difference]
        batch_norm_angles = np.array(batch_norm_angles)
        sorted_norm_angles = np.sort(batch_norm_angles)
        return sorted_norm_angles[0]


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
            from thirdparty.nn import nn_utils  # TODO: solve this reference
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
        def cal_one_add(pose_pred, pose_targets):
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
            return mean_dist
        
        if len(pose_pred.shape) == 2:
            mean_dist = cal_one_add(pose_pred, pose_targets)
        else:
            all_dists = []
            for idx in range(len(pose_pred)):
                one_dist = cal_one_add(pose_pred[idx], pose_targets[idx])
                all_dists.append(one_dist)
            sorted_dists = np.sort(all_dists)
            mean_dist = sorted_dists[0]
        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)
        return mean_dist, mean_dist < diameter

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
        print('2d projections metric: {}'.format(proj2d * 100))
        print('ADD metric: {}'.format(add * 100))
        print('ADD2 metric: {}'.format(add2 * 100))
        print('ADD5 metric: {}'.format(add5 * 100))
        # print('ADDS metric: {}'.format(adds))
        print('5 cm 5 degree metric: {}'.format(cmd5 * 100))
        # print('mask ap70: {}'.format(ap))
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
        
    def evaluate_proposals(self, pose_gt, pose_proposals, cam_k):
        # copy the gt to batch form
        proposal_num = len(pose_proposals)
        pose_gt = np.array([pose_gt for i in range(proposal_num)])
        ang_err_euler = rotation_angle_euler(pose_gt[:, :3, :3], pose_proposals[:, :3, :3])
        trans_diff = [np.linalg.norm(pose_proposals[idx][:3, -1:] - pose_gt[idx][:3, -1:]) for idx in range(proposal_num)]
        trans_diff = np.sort(trans_diff)
        trans_err = np.min(trans_diff)
        if self.class_name in ['eggbox', 'glue']:
            add_value, add_final = self.add_metric(pose_proposals, pose_gt, syn=True)
        else:
            add_value, add_final = self.add_metric(pose_proposals, pose_gt)

        return {
            "ang_err_euler": ang_err_euler,
            "trans_err": trans_err,
            "add_value": add_value,
            "add_final": add_final
        }
    
    def evaluate_linemod(self, pose_gt, pose_pred, cam_k): # sample_correspondence_pairs=False, direct_align=False, use_cnnpose=True):
        ang_err_chordal = rotation_angle_chordal(pose_gt[:3, :3], pose_pred[:3, :3])
        ang_err_euler = rotation_angle_euler(pose_gt[:3, :3], pose_pred[:3, :3])
        trans_err = np.linalg.norm(pose_pred[:3, -1:] - pose_gt[:3, -1:])  # 3x1
        if self.class_name in ['eggbox', 'glue']:
            add_value, add_final = self.add_metric(pose_pred, pose_gt, syn=True)
            self.add2_metric(pose_pred, pose_gt, syn=True)
            self.add5_metric(pose_pred, pose_gt, syn=True)
        else:
            add_value, add_final = self.add_metric(pose_pred, pose_gt)
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
            "ang_err_chordal": ang_err_chordal,
            "ang_err_euler": ang_err_euler,
            "trans_err": trans_err,
            "pnp_inliers": -1,#len(inliers),
            "add_value": add_value,
            "add_final": add_final
            # "pc_proj_vis": pc_proj_vis,
            # "pc_proj_vis_pred": pc_proj_vis_pred,
            # "keypoints_2d_vis": np.zeros_like(pc_proj_vis_pred) #keypoints_2d_vis
        }
