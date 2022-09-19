import torch
import os
import numpy as np
from yono.load_everything import load_existing_model


def run_export_coarse(args, cfg, device):
    print('Export coarse visualization...')
    with torch.no_grad():
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
        # model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
        model, _, _ = load_existing_model(args, cfg, cfg.fine_train, ckpt_path, device=device)
        model.to(device)
        alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
        rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
    np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
    print('done')