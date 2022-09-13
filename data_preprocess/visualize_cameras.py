# This code is originated from [nerf plus plus](https://github.com/Kai-46/nerfplusplus)
import open3d as o3d
import json
import numpy as np
import cv2
import os
import pdb
import argparse
from vis import create_spheric_poses

np.random.seed(11)


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0.0, 1.0, 0.0]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2.0 / K[0, 0]) * 2.0)
    vfov = np.rad2deg(np.arctan(H / 2.0 / K[1, 1]) * 2.0)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.0))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.0))

    # build view frustum for camera (I, 0)
    frustum_points = np.array(
        [
            [0.0, 0.0, 0.0],  # frustum origin
            [-half_w, -half_h, frustum_length],  # top-left image corner
            [half_w, -half_h, frustum_length],  # top-right image corner
            [half_w, half_h, frustum_length],  # bottom-right image corner
            [-half_w, half_h, frustum_length],
        ]
    )  # bottom-left image corner
    frustum_lines = np.array(
        [[0, i] for i in range(1, 5)] + [[i, (i + 1)] for i in range(1, 4)] + [[4, 1]]
    )
    frustum_colors = np.tile(
        np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1)
    )

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(
        np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T
    )
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 5, 3))  # 5 vertices per frustum
    merged_lines = np.zeros((N * 8, 2))  # 8 lines per frustum
    merged_colors = np.zeros((N * 8, 3))  # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 5 : (i + 1) * 5, :] = frustum_points
        merged_lines[i * 8 : (i + 1) * 8, :] = frustum_lines + i * 5
        merged_colors[i * 8 : (i + 1) * 8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def visualize_cameras(
    colored_camera_dicts, sphere_radius, geometry_file=None, geometry_type="mesh"
):
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=sphere_radius, resolution=10
    )
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0.0, 0.0, 0.0]
    )
    things_to_draw = [sphere, coord_frame]

    idx = 0
    for color, camera_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]["K"]).reshape((4, 4))
            W2C = np.array(camera_dict[img_name]["W2C"]).reshape((4, 4))
            img_size = camera_dict[img_name]["img_size"]
            camera_size = camera_dict[img_name]["camera_size"]
            frustums.append(
                get_camera_frustum(
                    img_size, K, W2C, frustum_length=camera_size, color=color
                )
            )
            cnt += 1
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == "mesh":
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == "pointcloud":
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception("Unknown geometry_type: ", geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def read_single_scale_cam(path):
    with open(path, "r") as fp:
        meta = json.load(fp)
    frame = meta["frames"][0]
    fname = os.path.join(
        os.path.join(*path.split("/")[:-1]), frame["file_path"] + ".png"
    )
    if not fname.startswith("/"):
        fname = "/" + fname
    h, w, _ = cv2.imread(fname, -1).shape
    focal = 0.5 * w / np.tan(0.5 * float(meta["camera_angle_x"]))
    K = [[focal, 0, w / 2, 0], [0, focal, h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    cams_dict = {}
    z = []
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        C2W = np.array(frame["transform_matrix"], dtype=np.float32)
        # convert to right-down-forward coordinate
        z.append(C2W[2, 3])
        C2W = convert_pose(C2W)
        W2C = np.eye(4)
        W2C[:3, :3] = C2W[:3, :3].T
        W2C[:3, 3] = -C2W[:3, :3].T @ C2W[:3, 3]
        cams_dict[frame["file_path"].split("/")[-1]] = {
            "K": K,
            "W2C": W2C.flatten(),
            "img_size": [w, h],
            "camera_size": 1,
        }
    return cams_dict, np.percentile(z, 90)


def read_multi_scale_cam(path):
    splits = ["train", "val", "test"]
    with open(path, "r") as fp:
        meta = json.load(fp)
    colored_camera_dicts = []
    render_color = np.random.rand(3)
    for split in splits:
        file_paths = meta[split]["file_path"]
        c2ws = np.array(meta[split]["cam2world"])
        focals = np.array(meta[split]["focal"])
        widths = np.array(meta[split]["width"])
        heights = np.array(meta[split]["height"])
        color = np.random.rand(3)
        camera_size = 1
        for i in range(4):
            file_path = file_paths[i::4]
            c2w = c2ws[i::4]
            focal = focals[i::4]
            width = widths[i::4]
            height = heights[i::4]
            K = [
                [focal[0], 0, width[0] / 2, 0],
                [0, focal[0], height[0] / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
            cams_dict = {}
            z = []
            for j in range(len(file_path)):
                C2W = c2w[j]
                z.append(C2W[2, 3])
                C2W = convert_pose(C2W)
                W2C = np.eye(4)
                W2C[:3, :3] = C2W[:3, :3].T
                W2C[:3, 3] = -C2W[:3, :3].T @ C2W[:3, 3]
                cams_dict[file_path[j]] = {
                    "K": K,
                    "W2C": W2C.flatten(),
                    "img_size": [width[j], height[j]],
                    "camera_size": camera_size,
                }
            if split == "train":
                colored_camera_dicts.append(
                    (
                        render_color,
                        create_spheric_cam(
                            focal[0],
                            [width[0], height[0]],
                            np.percentile(z, 90),
                            camera_size,
                        ),
                    )
                )
            camera_size /= 2
            colored_camera_dicts.append((color, cams_dict))

    return colored_camera_dicts


def create_spheric_cam(focal, img_size, radius, camera_size=1):
    w, h = img_size
    K = [[focal, 0, w / 2, 0], [0, focal, h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    cams_dict = {}
    C2Ws = create_spheric_poses(radius)
    for i in range(len(C2Ws)):
        C2W = C2Ws[i, ...]
        # convert to right-down-forward coordinate
        C2W = convert_pose(C2W)
        W2C = np.eye(4)
        W2C[:3, :3] = C2W[:3, :3].T
        W2C[:3, 3] = -C2W[:3, :3].T @ C2W[:3, 3]
        cams_dict[str(i)] = {
            "K": K,
            "W2C": W2C.flatten(),
            "img_size": [w, h],
            "camera_size": camera_size,
        }
    return cams_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Path to data.",
    )
    parser.add_argument(
        "--multi_scale", help="Whether vis multi camera.", action="store_true"
    )
    args = parser.parse_args()
    sphere_radius = 1.0
    if not args.multi_scale:
        train_cam_dict, radius = read_single_scale_cam(
            os.path.join(args.data_path, "transforms_train.json")
        )
        val_cam_dict, _ = read_single_scale_cam(
            os.path.join(args.data_path, "transforms_val.json")
        )
        test_cam_dict, _ = read_single_scale_cam(
            os.path.join(args.data_path, "transforms_test.json")
        )
        cam = train_cam_dict[list(train_cam_dict.keys())[0]]
        render_dict = create_spheric_cam(cam["K"][0][0], cam["img_size"], radius)
        colored_camera_dicts = [
            ([0, 1, 0], train_cam_dict),
            ([0, 0, 1], val_cam_dict),
            ([1, 1, 0], test_cam_dict),
            ([0, 1, 1], render_dict),
        ]
    else:
        colored_camera_dicts = read_multi_scale_cam(
            os.path.join(args.data_path, "metadata.json")
        )

    visualize_cameras(colored_camera_dicts, sphere_radius)
