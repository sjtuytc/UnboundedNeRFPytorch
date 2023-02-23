import argparse
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path')
parser.add_argument('thres', type=float)
parser.add_argument('--cam')
args = parser.parse_args()

data = np.load(args.path)
alpha = data['alpha']
rgb = data['rgb']
if rgb.shape[0] < rgb.shape[-1]:
    alpha = np.transpose(alpha, (1,2,0))
    rgb = np.transpose(rgb, (1,2,3,0))
print('Shape', alpha.shape, rgb.shape)
print('Active rate', (alpha > args.thres).mean())
print('Active nums', (alpha > args.thres).sum())
xyz_min = np.array([0,0,0])
xyz_max = np.array(alpha.shape)

if args.cam:
    data = np.load(args.cam)
    xyz_min = data['xyz_min']
    xyz_max = data['xyz_max']
    cam_lst = data['cam_lst']
    cam_frustrm_lst = []
    for cam in cam_lst:
        cam_frustrm = o3d.geometry.LineSet()
        cam_frustrm.points = o3d.utility.Vector3dVector(cam)
        if len(cam) == 5:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5] for i in range(8)])
            cam_frustrm.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,4],[4,3],[3,1]])
        elif len(cam) == 8:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5] for i in range(12)])
            cam_frustrm.lines = o3d.utility.Vector2iVector([
                [0,1],[1,3],[3,2],[2,0],
                [4,5],[5,7],[7,6],[6,4],
                [0,4],[1,5],[3,7],[2,6],
            ])
        cam_frustrm_lst.append(cam_frustrm)
else:
    cam_frustrm_lst = []


aabb_01 = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0]])
out_bbox = o3d.geometry.LineSet()
out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
out_bbox.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
out_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

xyz = np.stack((alpha > args.thres).nonzero(), -1)
color = rgb[xyz[:,0], xyz[:,1], xyz[:,2]]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz / alpha.shape * (xyz_max - xyz_min) + xyz_min)
pcd.colors = o3d.utility.Vector3dVector(color[:, :3])
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=max((xyz_max - xyz_min) / alpha.shape))

def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

o3d.visualization.draw_geometries_with_key_callbacks([
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=(xyz_max-xyz_min).min()*0.1, origin=xyz_min),
    out_bbox, voxel_grid, *cam_frustrm_lst,
], {ord("K"): change_background_to_black})

