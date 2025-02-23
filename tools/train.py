import os
import yaml
import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


from configs.config import CONF
from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset

from projects.model import RefHead, pl_model

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

#python /u/home/caoh/projects/MA_Jiachen/3DPNA/tools/train.py
def main():
    yaml_config = '/u/home/caoh/projects/MA_Jiachen/3DPNA/configs/KITTI_REF.yaml'

    with open(yaml_config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    

    dm = SemanticKITTIDataModule(
        dataset=SemanticKITTIDataset,
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model=config["data"]["pred_model"],
        )

    model = RefHead(
        num_class=config["model"]["num_class"],
        geo_feat_channels=config["model"]["embed_dim"],
        loss_weight_cfg=config["model"]["loss_weight_cfg"],
        balance_cls_weight=config["model"]["balance_cls_weight"],
        class_frequencies=CONF.semantic_kitti_class_frequencies
    )
    
    model = pl_model(
        model=model,
        config=config
    )

    log_folder = CONF.PATH.LOG_DIR

    check_path(log_folder)
    check_path(os.path.join(log_folder, 'tensorboard'))
    check_path(os.path.join(log_folder, 'csv_logs'))

    tb_logger = TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard'
    )
    
    csv_logger = CSVLogger(
        save_dir=log_folder,
        name='csv_logs'
    )
    
    logger = [tb_logger, csv_logger]

    version = tb_logger.version
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_folder, f'ckpts/version_{version}'),
        monitor='val/mIoU',
        mode='max',
        save_top_k=-1,
        save_last=False,
        filename='epoch_{epoch:03d}-mIoU={mIoU:.4f}'
        )

    # trainer
    trainer = pl.Trainer(
        devices=[i for i in range(num_gpu)],
        strategy=DDPStrategy(
            accelerator='gpu',
            find_unused_parameters=False
        ),
        max_steps=config["training"]['training_steps'],
        #resume_from_checkpoint=config['load_from'],
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step')
        ],
        logger=logger,
        profiler="simple",
        sync_batchnorm=True,
        log_every_n_steps=config["training"]['log_every_n_steps'],
        check_val_every_n_epoch=config["training"]['check_val_every_n_epoch']
    )

    trainer.fit(model=model, datamodule=dm)
    
if __name__ == "__main__":
    main()

