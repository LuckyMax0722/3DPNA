import os
import torch
import numpy as np

import pytorch_lightning as pl

from projects.loss import SSCMetrics
from projects.loss import get_inv_map


class pl_model_pna_text(pl.LightningModule):
    def __init__(
        self,
        model,
        model_version,
        config
        ):
        super(pl_model_pna_text, self).__init__()

        self.model = model
        self.model_version = model_version
        self.config = config

        self.num_class = config['model']['num_class']
        self.class_names = config['model']['class_names']

        self.train_metrics = SSCMetrics()
        self.val_metrics = SSCMetrics()
        self.test_metrics = SSCMetrics()

        self.save_path = config['model']['save_path']
        self.test_mapping = config['model']['test_mapping']
        self.pretrain = config['model']['pretrain']
        
    def forward_train(self, data_dict):
        gt_occ_256 = data_dict['gt_occ']  # [1, 256, 256, 32]
        gt_occ_128 = data_dict['gt_occ_2']  # [1, 128, 128, 16]
        gt_occ_64 = data_dict['gt_occ_4']  # [1, 64, 64, 8]
        gt_occ_32 = data_dict['gt_occ_8']  # [1, 32, 32, 4]

        input_occ = data_dict['input_occ'] # [1, 256, 256, 32]
        
        losses = dict()

        if self.model_version == 'pna_text':
            output_PNA, output_TEXT = self.model(input_occ , data_dict['text_feat'])

            losses_occupancy = self.model.ms_loss(
                output_voxels_list=output_PNA,
                target_voxels_list=[gt_occ_32, gt_occ_64, gt_occ_128, gt_occ_256],
                branch='PNA'
            )

            losses.update(losses_occupancy)

            losses_occupancy = self.model.ms_loss(
                output_voxels_list=output_TEXT,
                target_voxels_list=[gt_occ_32, gt_occ_64, gt_occ_128, gt_occ_256],
                branch='TEXT'
            )

            losses.update(losses_occupancy)

            losses_kl = self.model.kl_loss(
                pna_branch_list=output_PNA, 
                text_branch_list=output_TEXT
            )

            losses.update(losses_kl)

        pred = torch.argmax(output_PNA[-1], dim=1)
            
        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ_256
        }

        return train_output

    def forward_test(self, data_dict):
        input_occ = data_dict['input_occ'] # [1, 256, 256, 32]
        gt_occ_256 = data_dict['gt_occ']

        if self.model_version == 'pna_text':
            output_PNA, output_TEXT = self.model(input_occ , data_dict['text_feat'])

        pred = torch.argmax(output_PNA[-1], dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ_256
        }

        return test_output
        
    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

            
    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        loss_dict = output_dict['losses']
        loss = 0
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value.detach(), on_epoch=True, sync_dist=True)
            loss += value

        self.log("train/loss", loss.detach(), on_epoch=True, sync_dist=True, prog_bar=True)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach()
            gt_occ = output_dict['gt_occ'].detach()

            self.train_metrics.update(pred, gt_occ)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        output_dict = self.forward(batch)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach()
            gt_occ = output_dict['gt_occ'].detach()
            
            self.val_metrics.update(pred, gt_occ)
    
    def on_validation_epoch_end(self):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        # metric_list = [("val", self.val_metrics)]
        
        metrics_list = metric_list
        
        for prefix, metric in metrics_list:
            stats = metric.compute()

            if prefix == 'val':
                for name, iou in zip(self.class_names, stats['iou_ssc']):
                    #self.log("{}/{}/IoU".format(prefix, name), torch.tensor(iou, dtype=torch.float32).cuda(), sync_dist=True)
                    self.log(f"{prefix}/{name}/IoU", iou, sync_dist=True)

            self.log(f"{prefix}/mIoU", stats["iou_ssc_mean"], sync_dist=True)
            self.log(f"{prefix}/IoU", stats["iou"], sync_dist=True)
            self.log(f"{prefix}/Precision", stats["precision"], sync_dist=True)
            self.log(f"{prefix}/Recall", stats["recall"], sync_dist=True)

            metric.reset()
        
    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict['pred'].detach().cpu().numpy()
        gt_occ = output_dict['gt_occ']
        if gt_occ is not None:
            gt_occ = gt_occ.detach()
        else:
            gt_occ = None
            
        if self.save_path is not None:
            if self.test_mapping:
                inv_map = get_inv_map()
                output_voxels = inv_map[pred].astype(np.uint16)
            else:
                output_voxels = pred.astype(np.uint16)
            sequence_id = batch['img_metas']['sequence'][0]
            frame_id = batch['img_metas']['frame_id'][0]
            save_folder = "{}/sequences/{}/predictions".format(self.save_path, sequence_id)
            save_file = os.path.join(save_folder, "{}.label".format(frame_id))
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, 'wb') as f:
                output_voxels.tofile(f)
                print('\n save to {}'.format(save_file))
            
        if gt_occ is not None:
            self.test_metrics.update(pred, gt_occ)
    
    def on_test_epoch_end(self):
        metric_list = [("test", self.test_metrics)]
        # metric_list = [("val", self.val_metrics)]
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = metric.compute()

            for name, iou in zip(self.class_names, stats['iou_ssc']):
                #print(name + ":", iou)
                print(name + ":", iou)

            self.log(f"{prefix}/mIoU", stats["iou_ssc_mean"], sync_dist=True)
            self.log(f"{prefix}/IoU", stats["iou"], sync_dist=True)
            self.log(f"{prefix}/Precision", stats["precision"], sync_dist=True)
            self.log(f"{prefix}/Recall", stats["recall"], sync_dist=True)

            metric.reset()

    def configure_optimizers(self):
        if self.config['optimizer']['type'] == 'AdamW':

            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )

        else:
            raise NotImplementedError(f"Optimizer {self.config['optimizer']['type']} is not implemented.")
        
        if self.config['lr_scheduler']['type'] == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config['lr_scheduler']['max_lr'],
                total_steps=self.config['lr_scheduler']['total_steps'],
                pct_start=self.config['lr_scheduler']['pct_start'],
                cycle_momentum=self.config['lr_scheduler']['cycle_momentum'],
                anneal_strategy=self.config['lr_scheduler']['anneal_strategy'])

            interval=self.config['lr_scheduler']['interval']
            frequency=self.config['lr_scheduler']['frequency']
        else:
            raise NotImplementedError(f"lr_scheduler {self.config['lr_scheduler']['type']} is not implemented.")
        
        scheduler = {'scheduler': lr_scheduler, 'interval': interval, 'frequency': frequency}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    