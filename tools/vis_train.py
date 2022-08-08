import argparse
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path')
args = parser.parse_args()

data = np.load(args.path)
xyz_min = data['xyz_min']
xyz_max = data['xyz_max']
cam_lst = data['cam_lst']

# Outer aabb
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

# Cameras
cam_frustrm_lst = []
for cam in cam_lst:
    cam_frustrm = o3d.geometry.LineSet()
    cam_frustrm.points = o3d.utility.Vector3dVector(cam)
    if len(cam) == 5:
        cam_frustrm.colors = o3d.utility.Vector3dVector([[0,0,0] for i in range(8)])
        cam_frustrm.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,4],[4,3],[3,1]])
    elif len(cam) == 8:
        cam_frustrm.colors = o3d.utility.Vector3dVector([[0,0,0] for i in range(12)])
        cam_frustrm.lines = o3d.utility.Vector2iVector([
            [0,1],[1,3],[3,2],[2,0],
            [4,5],[5,7],[7,6],[6,4],
            [0,4],[1,5],[3,7],[2,6],
        ])
    else:
        raise NotImplementedError
    cam_frustrm_lst.append(cam_frustrm)

# Show
o3d.visualization.draw_geometries([
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=xyz_min),
    out_bbox, *cam_frustrm_lst])

