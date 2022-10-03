from yono.yono_model import YONOModel
from yono import utils, dvgo, dcvgo, dmpigo
import torch
import pdb


class YONOCheckpointManager:
    def __init__(self, args, cfg) -> None:
        super(YONOCheckpointManager, self).__init__()
        self.args = args
        self.cfg = cfg
    
    def load_model(self, model_class, ckpt_path):
        ckpt = torch.load(ckpt_path)
        model = model_class(**ckpt['model_kwargs'])
        model.load_state_dict(ckpt['model_state_dict'])
        return model
    
    def load_all_info(self, model, optimizer, ckpt_path, no_reload_optimizer):
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        model.load_state_dict(ckpt['model_state_dict'])
        if not no_reload_optimizer:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return model, optimizer, start

    def load_existing_model(self, args, cfg, cfg_train, reload_ckpt_path, device):
        # not used in training
        if cfg.data.dataset_type == "waymo":
            model_class = YONOModel
        elif cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model = self.load_model(model_class, reload_ckpt_path).to(device)
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


