from os.path import join as pjoin
from copy import deepcopy
from glob import glob
import click
import pdb
import numpy as np
import camera_utils
import cv2 as cv
from pycolmap.pycolmap.scene_manager import SceneManager
from typing import Mapping, Optional, Sequence, Text, Tuple, Union


# This implementation is from MipNeRF360
class NeRFSceneManager(SceneManager):
    """COLMAP pose loader.

    Minor NeRF-specific extension to the third_party Python COLMAP loader:
    google3/third_party/py/pycolmap/scene_manager.py
    """

    def __init__(self, data_dir):
        super(NeRFSceneManager, self).__init__(pjoin(data_dir, 'sparse', '0'))

    def process(
            self
    ) -> Tuple[Sequence[Text], np.ndarray, np.ndarray, Optional[Mapping[
        Text, float]], camera_utils.ProjectionType]:
        """Applies NeRF-specific postprocessing to the loaded pose data.

        Returns:
          a tuple [image_names, poses, pixtocam, distortion_params].
          image_names:  contains the only the basename of the images.
          poses: [N, 4, 4] array containing the camera to world matrices.
          pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
          distortion_params: mapping of distortion param name to distortion
            parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
        """

        self.load_cameras()
        self.load_images()
        self.load_points3D()

        # Assume shared intrinsics between all cameras.
        cam = self.cameras[1]

        # Extract focal lengths and principal point parameters.
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        c2w_mats = np.linalg.inv(w2c_mats)
        poses = c2w_mats[:, :3, :4]

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        names = [imdata[k].name for k in imdata]
        
        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        poses = poses @ np.diag([1, -1, -1, 1])
        # pixtocam = np.diag([1, -1, -1]) @ pixtocam

        # Get distortion parameters.
        type_ = cam.camera_type

        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 1 or type_ == 'PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        if type_ == 2 or type_ == 'SIMPLE_RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 3 or type_ == 'RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 4 or type_ == 'OPENCV':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['p1'] = cam.p1
            params['p2'] = cam.p2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['k3'] = cam.k3
            params['k4'] = cam.k4
            camtype = camera_utils.ProjectionType.FISHEYE

        return names, poses, pixtocam, params, camtype


class Dataset:
    def __init__(self, data_dir):
        scene_manager = NeRFSceneManager(data_dir)
        self.names, self.poses, self.pix2cam, self.params, self.camtype = scene_manager.process()
        self.cam2pix = np.linalg.inv(self.pix2cam)
        self.n_images = len(self.poses)

        # re-permute images by name
        sorted_image_names = sorted(deepcopy(self.names))
        sort_img_idx = []
        for i in range(self.n_images):
            sort_img_idx.append(self.names.index(sorted_image_names[i]))
        img_idx = np.array(sort_img_idx, dtype=np.int32)
        self.poses = self.poses[sort_img_idx]

        # calc near-far bounds
        self.bounds = np.zeros([self.n_images, 2], dtype=np.float32)
        name_to_ids = scene_manager.name_to_image_id
        points3D = scene_manager.points3D
        points3D_ids = scene_manager.point3D_ids
        point3D_id_to_images = scene_manager.point3D_id_to_images
        image_id_to_image_idx = np.zeros(self.n_images + 10, dtype=np.int32)
        for image_name in self.names:
            image_id_to_image_idx[name_to_ids[image_name]] = sorted_image_names.index(image_name)

        vis_arr = []
        for pts_i in range(len(points3D)):
            cams = np.zeros([self.n_images], dtype=np.uint8)
            images_ids = point3D_id_to_images[points3D_ids[pts_i]]
            for image_info in images_ids:
                image_id = image_info[0]
                image_idx = image_id_to_image_idx[image_id]
                cams[image_idx] = 1
            vis_arr.append(cams)

        vis_arr = np.stack(vis_arr, 1)     # [n_images, n_pts ]

        for img_i in range(self.n_images):
            vis = vis_arr[img_i]
            pts = points3D[vis == 1]
            c2w = np.diag([1., 1., 1., 1.])
            c2w[:3, :4] = self.poses[img_i]
            w2c = np.linalg.inv(c2w)
            z_vals = (w2c[None, 2, :3] * pts).sum(-1) + w2c[None, 2, 3]
            depth = -z_vals
            near_depth, far_depth = np.percentile(depth, 1.), np.percentile(depth, 99.)
            near_depth = near_depth * .5
            far_depth = far_depth * 5.
            self.bounds[img_i, 0], self.bounds[img_i, 1] = near_depth, far_depth

        # Move all to numpy
        def proc(x):
            return np.ascontiguousarray(np.array(x).astype(np.float64))

        self.poses = proc(self.poses)
        self.cam2pix = proc(np.tile(self.cam2pix[None], (len(self.poses), 1, 1)))
        self.bounds = proc(self.bounds)
        if self.params is not None:
            dist_params = [ self.params['k1'], self.params['k2'], self.params['p1'], self.params['p2'] ]
        else:
            dist_params = [0., 0., 0., 0.]
        dist_params = np.tile(np.array(dist_params), len(self.poses)).reshape([len(self.poses), -1])
        self.dist_params = proc([dist_params])

    def export(self, data_dir, out_mode):
        n = len(self.poses)
        if out_mode == 'cams_meta':
            data = np.concatenate([self.poses.reshape([n, -1]),
                                   self.cam2pix.reshape([n, -1]),
                                   self.dist_params.reshape([n, -1]),
                                   self.bounds.reshape([n, -1])], axis=-1)
            data = np.ascontiguousarray(np.array(data).astype(np.float64))
            np.save(pjoin(data_dir, 'cams_meta.npy'), data)
        elif 'poses_bounds' in out_mode :
            poses = deepcopy(self.poses)
            image_list = []
            suffs = ['*.png', '*.PNG', '*.jpg', '*.JPG']
            for suff in suffs:
                image_list += glob(pjoin(data_dir, 'images', suff))
            h, w, _ = cv.imread(image_list[0]).shape
            focal = (self.cam2pix[0, 0, 0] + self.cam2pix[0, 1, 1]) * .5

            # poses_ = torch::cat({ poses_.index({Slc(), Slc(), Slc(1, 2)}),
            # -poses_.index({Slc(), Slc(), Slc(0, 1)}),
            # poses_.index({Slc(), Slc(), Slc(2, None)})}, 2);

            if out_mode == 'poses_bounds_raw':
                poses = np.concatenate([-poses[:, :, 1:2], poses[:, :, 0:1], poses[:, :, 2:]], 2)

            hwf = np.zeros([n, 3])
            hwf[:, 0] = h
            hwf[:, 1] = w
            hwf[:, 2] = focal
            bounds = self.bounds
            poses_hwf = np.concatenate([poses, hwf[:, :, None]], -1)
            data = np.concatenate([poses_hwf.reshape([n, -1]), bounds.reshape([n, -1])], -1)
            data = np.ascontiguousarray(np.array(data).astype(np.float64))
            np.save(pjoin(data_dir, '{}.npy'.format(out_mode)), data)


@click.command()
@click.option('--data_dir', type=str)
@click.option('--out_mode', type=str, default='cams_meta')
def main(data_dir, out_mode):
    dataset = Dataset(data_dir)
    dataset.export(data_dir, out_mode)


if __name__ == '__main__':
    main()
