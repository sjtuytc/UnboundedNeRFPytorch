from yono.yono_model import YONOModel
from yono import utils, dvgo, dcvgo, dmpigo
import torch
import pdb
import os
from tqdm import tqdm
from yono.run_train import create_new_model
import torch.nn.functional as F


class YONOCheckpointManager:
    def __init__(self, args, cfg) -> None:
        super(YONOCheckpointManager, self).__init__()
        self.args = args
        self.cfg = cfg

    def load_all_info(self, model, optimizer, ckpt_path, no_reload_optimizer):
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        model.load_state_dict(ckpt['model_state_dict'])
        if not no_reload_optimizer:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return model, optimizer, start

    def load_existing_model(self, args, cfg, cfg_train, reload_ckpt_path, device):
        # not used in training
        if cfg.data.dataset_type == "waymo" or cfg.data.dataset_type == "mega":
            model_class = YONOModel
        elif cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model, _ = self.load_model(model_class, reload_ckpt_path)
        model = model.to(device)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
        model, optimizer, start = self.load_all_info(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
        return model, optimizer, start

    def save_model(self, global_step, model, optimizer, save_path):
        if self.args.block_num > 1:
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path.replace(".tar", f"_{self.args.running_block_id}.tar"))
        else:
            torch.save({
                    'global_step': global_step,
                    'model_kwargs': model.get_kwargs(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)

    def load_model(self, model_class, ckpt_path):
        ckpt = torch.load(ckpt_path)
        model_args = ckpt['model_kwargs']
        model = model_class(**model_args)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, model_args

    def merge_two_grid(self, grid_one, grid_two, merged_grid, center1, radius1, center2, radius2, ndc_xyz_min, ndc_xyz_max,
                       ):

        pdb.set_trace()
        # TODO: consider the background merging functions
        return merged_grid
    
    @torch.no_grad()
    def merge_blocks(self, ckpt_paths, device, model_class, exp_dir):
        # for idx, cp in enumerate(tqdm(ckpt_paths)):
        #     model = utils.load_model(model_class, cp).to(device)
        #     model.export_geometry_for_visualize(os.path.join(exp_dir, f"debug{idx}.npz"))
        cp1, cp2 = ckpt_paths[0], ckpt_paths[1]
        m1, m1_args = self.load_model(model_class, cp1)
        m1 = m1.to(device)
        m2, m2_args = self.load_model(model_class, cp2)
        m2 = m2.to(device)
        m1.export_geometry_for_visualize(os.path.join(exp_dir, "debug1.npz"))
        m2.export_geometry_for_visualize(os.path.join(exp_dir, "debug2.npz"))
        center1, center2 = m1.scene_center, m2.scene_center
        r1, r2 = m1.scene_radius, m2.scene_radius
        global_xyz_min_1, global_xyz_max_1, global_xyz_min_2, global_xyz_max_2 = center1 - r1, center1 + r1, \
            center2 - r2, center2 + r2
        
        # compose a new model, create a bigger model first
        global_xyz_min_merged = torch.tensor([torch.min(global_xyz_min_1[i], global_xyz_min_1[i]) for i in range(3)])
        # global_xyz_min_merged = torch.tensor([torch.min(global_xyz_min_1[i], global_xyz_min_2[i]) for i in range(3)])
        global_xyz_max_merged = torch.tensor([torch.max(global_xyz_max_1[i], global_xyz_max_1[i]) for i in range(3)])
        # global_xyz_max_merged = torch.tensor([torch.max(global_xyz_max_1[i], global_xyz_max_2[i]) for i in range(3)])
        merged_model_args = {}
        for m1_key in m1_args:
            if m1_key in ['xyz_min', 'xyz_max']:
                assert torch.equal(torch.tensor(m1_args[m1_key]), torch.tensor(m2_args[m1_key])), "Two models are of different configs."
                merged_model_args['xyz_min'] = global_xyz_min_merged
                merged_model_args['xyz_max'] = global_xyz_max_merged
            else:
                assert m1_args[m1_key] == m2_args[m1_key], "Two models are of different configs."
                merged_model_args[m1_key] = m1_args[m1_key]
        xyz_min_ndc, xyz_max_ndc = m1_args['xyz_min'], m1_args['xyz_max']  # shared for two blocks
        merged_model = model_class(**merged_model_args).to(device)
        
        # merge the grids
        merged_state_dict = {}
        for key in merged_model.state_dict():
            if key in ['scene_center', 'scene_radius', 'xyz_min', 'xyz_max', 'density.xyz_min', \
                'density.xyz_max', 'k0.xyz_min', 'k0.xyz_max', 'mask_cache.mask', ]:
                merged_state_dict[key] = merged_model.state_dict()[key]  # use the merged values or the same as the initial model
            elif key in ['act_shift', 'xyz2ijk_scale', 'mask_cache.xyz2ijk_shift', 'mask_cache.xyz2ijk_scale']:
                # the same as either models
                merged_state_dict[key] = m1.state_dict()[key]
            elif key in ['density.grid', 'k0.grid', ]:
                # needs to merge
                g1, g2 = m1.state_dict()[key], m2.state_dict()[key]
                m1_movements_in_merged = (m1.scene_center - merged_model.scene_center) / (global_xyz_max_merged - global_xyz_min_merged)
                m2_movements_in_merged = (m2.scene_center - merged_model.scene_center) / (global_xyz_max_merged - global_xyz_min_merged)
                # transform ndc to grid indexs
                m1_movements_in_merged = m1_movements_in_merged.flip(-1) * g1.shape[-1]
                m2_movements_in_merged = m2_movements_in_merged.flip(-1) * g2.shape[-1]
                
                # move two grids to merged grids
                merged_g = merged_model.state_dict()[key]
                g1_rolled = torch.roll(g1[0], shifts=tuple(m1_movements_in_merged.long().cpu().numpy()), dims=[0, 1, 2])
                g2_rolled = torch.roll(g2[0], shifts=tuple(m2_movements_in_merged.long().cpu().numpy()), dims=[0, 1, 2])
                # merged_g[0] = (g1_rolled + g2_rolled) * 0.5
                merged_g[0] = g1
                merged_state_dict[key] = merged_g
                # ndc_xyz_min_1 = (global_xyz_min_1 - m1.scene_center) / m1.scene_radius
                # ndc_xyz_max_1 = (global_xyz_max_1 - m1.scene_center) / m1.scene_radius
                # ndc_xyz_min_2 = (global_xyz_min_2 - m2.scene_center) / m2.scene_radius
                # ndc_xyz_max_2 = (global_xyz_max_2 - m2.scene_center) / m2.scene_radius
                
                # # merge two grids
                # x = torch.tensor(list(range(merged_g.shape[-3])))
                # y = torch.tensor(list(range(merged_g.shape[-2])))
                # z = torch.tensor(list(range(merged_g.shape[-1])))
                # all_coords_in_new_ndc = torch.cartesian_prod(x, y, z)
                # global_coordinates_merged = global_xyz_min_merged + (global_xyz_max_merged - global_xyz_min_merged) * all_coords_in_new_ndc
                # ndc_coordinates_g1 = (global_coordinates_merged - m1.scene_center) / m1.scene_radius
                # ndc_coordinates_g2 = (global_coordinates_merged - m2.scene_center) / m2.scene_radius
                
                # # sample first grid
                # xyz_min_ndc = torch.tensor(xyz_min_ndc).to(device)
                # xyz_max_ndc = torch.tensor(xyz_max_ndc).to(device)
                # ndc_coords_cut_g1 = torch.clamp(ndc_coordinates_g1, min=torch.tensor(xyz_min_ndc).to(device), max=torch.tensor(xyz_max_ndc).to(device))
                # ndc_coords_cut_g1 = ndc_coords_cut_g1.to(device).reshape(1, 1, 1, -1, 3)
                # ndc_coords_cut_g2 = torch.clamp(ndc_coordinates_g2, min=torch.tensor(xyz_min_ndc).to(device), max=torch.tensor(xyz_max_ndc).to(device))
                # ndc_coords_cut_g2 = ndc_coords_cut_g2.to(device).reshape(1, 1, 1, -1, 3)
                # ind_norm_g1 = ((ndc_coords_cut_g1 - xyz_min_ndc) / (xyz_max_ndc - xyz_min_ndc)).flip((-1,)) * 2 - 1
                # ind_norm_g2 = ((ndc_coords_cut_g2 - xyz_min_ndc) / (xyz_max_ndc - xyz_min_ndc)).flip((-1,)) * 2 - 1
                
                # g1_sampled = F.grid_sample(g1, ind_norm_g1, mode='nearest', align_corners=True)
                # g2_sampled = F.grid_sample(g2, ind_norm_g2, mode='bilinear', align_corners=True)
                # merged_g1 = g1_sampled.reshape(merged_g.shape)
                # merged_g2 = g2_sampled.reshape(merged_g.shape)
                # merged_g_final = merged_g1 + merged_g2
                # merged_state_dict[key] = merged_g_final
            else:
                pdb.set_trace()
                raise KeyError("Not supported keys.")
        merged_model.load_state_dict(merged_state_dict)
        merged_model.update_occupancy_cache()
        merged_model.export_geometry_for_visualize(os.path.join(exp_dir, "debug.npz"))
        
        return merged_model
