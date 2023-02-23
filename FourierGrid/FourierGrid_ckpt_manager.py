from FourierGrid.FourierGrid_model import FourierGridModel
from FourierGrid import utils, dvgo, dcvgo, dmpigo
import torch
import pdb
import os
from tqdm import tqdm
from FourierGrid.run_train import create_new_model
import torch.nn.functional as F


class FourierGridCheckpointManager:
    def __init__(self, args, cfg) -> None:
        super(FourierGridCheckpointManager, self).__init__()
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
        FourierGrid_datasets = ["waymo", "mega", "nerfpp", "tankstemple"]
        if cfg.data.dataset_type in FourierGrid_datasets or cfg.model == 'FourierGrid':
            model_class = FourierGridModel
        elif cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model, _ = self.load_model(model_class, reload_ckpt_path)
        model = model.to(device)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0,
                                                           verbose=False)
        model, optimizer, start = self.load_all_info(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
        return model, optimizer, start

    def save_model(self, global_step, model, optimizer, save_path):
        torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
        print(f'Saved checkpoints at', save_path)

    def load_model(self, model_class, ckpt_path):
        ckpt = torch.load(ckpt_path)
        model_args = ckpt['model_kwargs']
        model = model_class(**model_args)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, model_args
    
    @torch.no_grad()
    def merge_blocks(self, args, cfg, device):
        stage = 'fine'
        exp_folder = os.path.join(cfg.basedir, cfg.expname)
        paths = [os.path.join(exp_folder, f'{stage}_last_{block_id}.tar') for block_id in range(args.block_num)]
        model_class = FourierGridModel
        cp1 = paths[0]
        m1, m1_args = self.load_model(model_class, cp1)
        m1 = m1.to(device)
        
        merged_model, _ = create_new_model(args, cfg, cfg.fine_model_and_render, cfg.fine_train, 
                                           m1_args['xyz_min'], m1_args['xyz_max'], stage, None, device)
        merged_model = merged_model.to(device)
        merged_state_dict = m1.state_dict()
        # merge the grids consequently
        for idx, cur_cp in enumerate(paths[1:]):
            print(f"Meging grid {idx} / {len(paths[1:])}: {cur_cp} ...")
            cur_m, _ = self.load_model(model_class, cur_cp)
            cur_m = cur_m.to(device)
            for key in merged_state_dict:
                print(f"Merging model key: {key} ...")
                if key in ['density.grid', 'k0.grid'] or 'rgb' in key:
                    g1, g2 = merged_state_dict[key], cur_m.state_dict()[key]
                    merged_g = torch.min(g1, g2)
                    # merged_g = torch.max(g1, g2)
                    # del g1
                    # del g2
                    merged_state_dict[key] = merged_g
                # else:
                #     merged_state_dict[key] = merged_model.state_dict()[key]
            # del cur_m
            torch.cuda.empty_cache()
        if "mask_cache.mask" in merged_state_dict:
            merged_state_dict.pop("mask_cache.mask")
        merged_model.load_state_dict(merged_state_dict, strict=False)
        merged_model.update_occupancy_cache()
        # merged_model.export_geometry_for_visualize(os.path.join(exp_folder, "debug.npz"))
        return merged_model
