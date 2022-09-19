import numpy as np
import os
from pathlib import Path
from yono import utils, dvgo, dcvgo, dmpigo
from yono.bbox_compute import compute_bbox_by_cam_frustrm


def run_export_bbox_cams(args, cfg, data_dict): 
    print('Export bbox and cameras...')
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
    near, far = data_dict['near'], data_dict['far']
    if data_dict['near_clip'] is not None:
        near = data_dict['near_clip']
    cam_lst = []
    for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
        cam_o = rays_o[0,0].cpu().numpy()
        cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
        cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
    dir_name = os.path.dirname(args.export_bbox_and_cams_only)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.export_bbox_and_cams_only,
        xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
        cam_lst=np.array(cam_lst))
    print('done')
