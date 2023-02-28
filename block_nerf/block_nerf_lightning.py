from pytorch_lightning import LightningModule, Trainer
import torch
import os 
from collections import defaultdict
from torch.utils.data import DataLoader
from block_nerf.waymo_dataset import *
from block_nerf.block_nerf_model import *
from block_nerf.rendering import *
from block_nerf.metrics import *
from block_nerf.block_visualize import *
from block_nerf.learning_utils import *

class Block_NeRF_System(LightningModule):
    def __init__(self, hparams):
        super(Block_NeRF_System, self).__init__()
        self.hyper_params = hparams
        self.save_hyperparameters(hparams)
        self.loss = BlockNeRFLoss(1e-2)  #hparams['Visi_loss']

        self.xyz_IPE = InterPosEmbedding(hparams['N_IPE_xyz'])  # xyz的L=10
        self.dir_exposure_PE = PosEmbedding(
            hparams['N_PE_dir_exposure'])  # dir的L=4
        self.embedding_appearance = torch.nn.Embedding(
            hparams['N_vocab'], hparams['N_appearance'])

        self.Embedding = {'IPE': self.xyz_IPE,
                          'PE': self.dir_exposure_PE,
                          'appearance': self.embedding_appearance}

        self.Block_NeRF = Block_NeRF(in_channel_xyz=6 * hparams['N_IPE_xyz'],
                                     in_channel_dir=6 *
                                                    hparams['N_PE_dir_exposure'],
                                     in_channel_exposure=2 *
                                                         hparams['N_PE_dir_exposure'],
                                     in_channel_appearance=hparams['N_appearance'])

        self.Visibility = Visibility(in_channel_xyz=6 * hparams['N_IPE_xyz'],
                                     in_channel_dir=6 * hparams['N_PE_dir_exposure'])

        self.models_to_train = []
        self.models_to_train += [self.embedding_appearance]
        self.models_to_train += [self.Block_NeRF]
        self.models_to_train += [self.Visibility]

    def forward(self, rays, ts):
        B = rays.shape[0]
        model = {
            "block_model": self.Block_NeRF,
            "visibility_model": self.Visibility
        }

        results = defaultdict(list)
        for i in range(0, B, self.hparams['chunk']):
            rendered_ray_chunks = render_rays(model, self.Embedding,
                                              rays[i:i + self.hparams['chunk']],
                                              ts[i:i + self.hparams['chunk']],
                                              N_samples=self.hparams['N_samples'],
                                              N_importance=self.hparams['N_importance'],
                                              chunk=self.hparams['chunk'],
                                              type="train",
                                              use_disp=self.hparams['use_disp']
                                              )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    def setup(self, stage):
        self.train_dataset = WaymoDataset(root_dir=self.hparams['root_dir'],
                                          split='train',
                                          block=self.hparams['block_index'],
                                          img_downscale=self.hparams['img_downscale'],
                                          near=self.hparams['near'],
                                          far=self.hparams['far'])
        self.val_dataset = WaymoDataset(root_dir=self.hparams['root_dir'],
                                        split='val',
                                        block=self.hparams['block_index'],
                                        img_downscale=self.hparams['img_downscale'],
                                        near=self.hparams['near'],
                                        far=self.hparams['far'])

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams['batch_size'],
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=8,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            psnr_ = psnr(results['rgb_fine'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):  # validate at each epoch
        rays, rgbs, ts = batch['rays'].squeeze(), batch['rgbs'].squeeze(), batch['ts'].squeeze()
        W,H=batch['w_h']
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        if batch_nb == 0:
            img = results[f'rgb_fine'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_fine'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            #stack = torch.stack([img_gt, img])  # (3, 3, H, W)
            # todo: recheck this, * 255?
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        psnr_ = psnr(results['rgb_fine'], rgbs)

        log = {'val_loss': loss}
        for k, v in loss_d.items():
            log[f'val_{k}']= v
        log['val_psnr']= psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)