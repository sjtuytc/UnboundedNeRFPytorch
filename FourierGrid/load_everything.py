import torch
import pdb
from FourierGrid.common_data_loaders.load_common_data import load_common_data
from FourierGrid.load_waymo import load_waymo_data
from FourierGrid.load_mega import load_mega_data
# from FourierGrid.load_linemod import load_linemod_data
from FourierGrid import utils, dvgo, dcvgo, dmpigo
from FourierGrid.FourierGrid_model import FourierGridModel


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    if cfg.data.dataset_type == "waymo":
        data_dict = load_waymo_data(args, cfg)
        return data_dict, args
    elif cfg.data.dataset_type == "mega":
        data_dict = load_mega_data(args, cfg)
        return data_dict, args
    elif cfg.data.dataset_type == 'linemod':
        data_dict = load_linemod_data(args, cfg)
        return data_dict, args
    else:
        data_dict = load_common_data(cfg.data)
    
    # remove useless field
    kept_keys = {'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    if args.sample_num > 0:
        data_dict['i_train'] = data_dict['i_train'][:args.sample_num]
    else:
        args.sample_num = len(data_dict['i_train'])
    return data_dict, args


def load_existing_model(args, cfg, cfg_train, reload_ckpt_path, device):
    FourierGrid_datasets = ["waymo", "mega", "nerfpp", "tankstemple"]
    if cfg.data.dataset_type or cfg.model == 'FourierGrid':
        model_class = FourierGridModel
    elif cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start
