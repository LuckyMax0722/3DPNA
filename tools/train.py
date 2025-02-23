import os
import yaml
import numpy as np
import importlib.util

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


from configs.config import CONF
from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset

from projects.model import RefHead, pl_model, RefHead_PNA

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    return config_module.config

# python /u/home/caoh/projects/MA_Jiachen/3DPNA/tools/train.py

def main():
    version = 'pna'
    debug = True

    if debug:
        log_folder = '/u/home/caoh/projects/MA_Jiachen/3DPNA/a_tmp'
    else:
        log_folder = path_log_dir

    if version == 'small' or 'pna':
        yaml_config = os.path.join(CONF.PATH.CONFIG_DIR, 'REF_CGFormer.yaml')
    

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

    if version == 'small':
        model = RefHead(
            num_class=config["model"]["num_class"],
            geo_feat_channels=config["model"]["embed_dim"],
            loss_weight_cfg=config["model"]["loss_weight_cfg"],
            balance_cls_weight=config["model"]["balance_cls_weight"],
            class_frequencies=CONF.semantic_kitti_class_frequencies,
            
            skip_version='plus',
            conv_version='v1',
            head_version='conv'
        )

    elif version == 'pna':
        ffn_cfg=dict(
            type='FFN',
            embed_dims=config["model"]["embed_dim"],
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.1,
            add_identity=True
        )

        kernel_size = [[3,3,3], [5,5,5], [7,7,7]]
        dilation = [[1,1,1], [1,1,1], [1,1,1]]

        model = RefHead_PNA(
            num_class=config["model"]["num_class"],
            geo_feat_channels=config["model"]["embed_dim"],
            loss_weight_cfg=config["model"]["loss_weight_cfg"],
            balance_cls_weight=config["model"]["balance_cls_weight"],
            class_frequencies=CONF.semantic_kitti_class_frequencies,

            skip_version='plus',
            conv_version='v1',
            head_version='conv'


            ffn_cfg=ffn_cfg,
            num_heads=8,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=True,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1,
            use_fna=False,
        )
    
    model = pl_model(
        model=model,
        config=config
    )

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
        filename='epoch_{epoch:03d}-mIoU={val/mIoU:.4f}'
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

