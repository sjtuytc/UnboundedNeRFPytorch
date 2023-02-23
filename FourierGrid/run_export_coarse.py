import torch
import os
import numpy as np
from FourierGrid.load_everything import load_existing_model


def run_export_coarse(args, cfg, device, save_path=None):
    verbose = args.block_num <= 1
    if verbose:
        print('Export coarse visualization...')
    with torch.no_grad():
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
        # model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
        model, _, _ = load_existing_model(args, cfg, cfg.fine_train, ckpt_path, device=device)
        model.to(device)
        alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
        rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
    if save_path is None:
        save_path = args.export_coarse_only
    np.savez_compressed(save_path, alpha=alpha, rgb=rgb)
