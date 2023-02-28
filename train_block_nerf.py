import os
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from block_nerf.waymo_dataset import *
from block_nerf.block_nerf_model import *
from block_nerf.block_nerf_lightning import *
from block_nerf.rendering import *
from block_nerf.metrics import *

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='data/pytorch_waymo_dataset',
                        help='root directory of dataset')
    parser.add_argument('--block_index', type=int, default='0',
                        help='index of the blocks')
    parser.add_argument('--img_downscale', type=int, default=4,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--near', type=float, default=1e-2,
                        help='the range to sample along the ray')
    parser.add_argument('--far', type=float, default=15,
                        help='the range to sample along the ray')
    parser.add_argument('--N_IPE_xyz', type=int, default=16,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_PE_dir_exposure', type=int, default=4,
                        help='number of direction embedding frequencies'
                        )
    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')

    # NeRF-W

    parser.add_argument('--N_vocab', type=int, default=1500,
                        help='''number of vocabulary (number of images) 
                                        in the dataset for nn.Embedding'''
                        )
    parser.add_argument('--N_appearance', type=int, default=32,
                        help='number of embeddings for appearance')

    parser.add_argument('--Visi_loss', type=float, default=1e-2,
                        help='number of embeddings for appearance')

    parser.add_argument('--use_disp', type=bool, default=True,
                        help='use disparity depth sampling')

    parser.add_argument('--chunk', type=int, default=1024 * 16,
                        help='chunk to avoid OOM')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type', choices=['sgd', 'adam',
                        'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type', choices=['steplr',
                        'cosine', 'poly'])
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs'
                        )
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer'
                        )
    parser.add_argument('--decay_step', nargs='+', type=int,
                        default=[20], help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay'
                        )
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--refresh_every', type=int, default=1,
                        help='print the progress bar every X steps')

    return vars(parser.parse_args())


def main(hparams):
    print("Warning, this old implementation of BlockNeRF will be deprecated in the next version!")
    hparams['block_index'] = 'block_' + str(hparams['block_index'])
    system = Block_NeRF_System(hparams)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join('data/ckpts/{0}'.format(hparams['exp_name'
                        ]), str(hparams['block_index']) + '_{epoch:d}'
                        ), monitor='val/loss', mode='min', save_top_k=5)
    callbacks = [checkpoint_callback]

    logger = TensorBoardLogger(save_dir='logs',
                               name=hparams['block_index'],
                               default_hp_metric=False)

    trainer = Trainer(
        max_epochs=hparams['num_epochs'],
        precision=16,
        callbacks=callbacks,
        resume_from_checkpoint=hparams['ckpt_path'],
        logger=logger,
        enable_model_summary=True,
        gpus=hparams['num_gpus'],
        accelerator='auto',
        num_sanity_val_steps=1,
        benchmark=True,
        profiler=('simple' if hparams['num_gpus'] == 1 else None),
        strategy=(DDPPlugin(find_unused_parameters=False) if hparams['num_gpus'
                  ] > 1 else None),
        )

    trainer.fit(system)
    print('The best model is saved in', checkpoint_callback.best_model_path)


if __name__ == '__main__':
    hparams = get_opts()
    torch.cuda.empty_cache()
    main(hparams)
