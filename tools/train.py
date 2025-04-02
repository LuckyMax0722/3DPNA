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

from projects.model import RefHead, pl_model, pl_model_diff, pl_model_lseg, RefHead_PNA, RefHead_VQ, RefHead_CVAE
from projects.model import RefHead_TPV_Lseg, RefHead_TPV_Lseg_V2, RefHead_TPV_Lseg_V3, RefHead_TPV_Lseg_V3_FS
from projects.model import RefHead_Text, pl_model_text

from projects.model.diffusion import RefHead_D

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
    model_version = 'text'  # small, pna, vqvae cvae, diffusion, lseg
    baseline_model = 'CGFormer'  # CGFormer, MonoScene
    debug = True

    skip_version='concat'  # plus, concat, none
    encoder_version='conv'  # conv, aspp
    conv_version='v2'   # v1, v2
    head_version='conv'     # mlp, conv

    use_skip=True


    if debug:
        log_folder = '/u/home/caoh/projects/MA_Jiachen/3DPNA/a_tmp'
    else:
        log_folder = CONF.PATH.LOG_DIR

    yaml_config = os.path.join(CONF.PATH.CONFIG_DIR, ('REF_' + baseline_model + '.yaml'))

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

    if model_version == 'small':
        model = RefHead(
            num_class=config["model"]["num_class"],
            geo_feat_channels=config["model"]["embed_dim"],
            loss_weight_cfg=config["model"]["loss_weight_cfg"],
            balance_cls_weight=config["model"]["balance_cls_weight"],
            class_frequencies=CONF.semantic_kitti_class_frequencies,
            
            skip_version=skip_version,
            encoder_version=encoder_version,
            conv_version=conv_version,
            head_version=head_version,

            use_skip=use_skip
        )

    elif model_version == 'cvae':
        model = RefHead_CVAE(
            num_class=config["model"]["num_class"],
            geo_feat_channels=config["model"]["embed_dim"],
            loss_weight_cfg=config["model"]["loss_weight_cfg"],
            balance_cls_weight=config["model"]["balance_cls_weight"],
            class_frequencies=CONF.semantic_kitti_class_frequencies,
            
            skip_version=skip_version,
            encoder_version=encoder_version,
            conv_version=conv_version,
            head_version=head_version,

            use_skip=use_skip,
            latent_dim=256
        )

    elif model_version == 'pna':
        ffn_cfg=dict(
            type='FFN',
            embed_dims=config["model"]["embed_dim"],
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.1,
            add_identity=True
        )

        kernel_size = [[3,3,3], [3,3,3], [5,5,5]]
        dilation = [[1,1,1], [1,1,1], [1,1,1]]

        model = RefHead_PNA(
            num_class=config["model"]["num_class"],
            geo_feat_channels=config["model"]["embed_dim"],
            loss_weight_cfg=config["model"]["loss_weight_cfg"],
            balance_cls_weight=config["model"]["balance_cls_weight"],
            class_frequencies=CONF.semantic_kitti_class_frequencies,

            skip_version=skip_version,
            conv_version=conv_version,
            head_version=head_version,


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

    elif model_version == 'vqvae':
        model = RefHead_VQ(
            num_class = config["model"]["num_class"],
            init_size = 32,
            l_size = '882',
            l_attention = True,
            vq_size = 50,
            loss_weight_cfg=config["model"]["loss_weight_cfg"],
            balance_cls_weight=config["model"]["balance_cls_weight"],
            class_frequencies=CONF.semantic_kitti_class_frequencies,
        )

    elif model_version == 'diffusion':
        model = RefHead_D()

    elif model_version == 'lseg':
        ffn_cfg=dict(
            type='FFN',
            embed_dims=32,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.1,
            add_identity=True
        )

        TPV_version = 'fs'

        if TPV_version == 'v1':
            model = RefHead_TPV_Lseg(
                num_class=config["model"]["num_class"],
                geo_feat_channels=32,
                img_feat_channels=2048,
                kv_dim=256,
                z_down=True,

                dim_head=16,
                heads=2,
                ffn_cfg=ffn_cfg,

                loss_weight_cfg=config["model"]["loss_weight_cfg"],
                balance_cls_weight=config["model"]["balance_cls_weight"],
                class_frequencies=CONF.semantic_kitti_class_frequencies,
            )

            model = pl_model_lseg(
                model=model,
                model_version=model_version,
                TPV_version=TPV_version,
                config=config
            )


        elif TPV_version == 'v2':
            model = RefHead_TPV_Lseg_V2(
                num_class=config["model"]["num_class"],
                geo_feat_channels=32,
                img_feat_channels=2048,
                kv_dim=256,
                z_down=True,

                dim_head=16,
                heads=2,
                ffn_cfg=ffn_cfg,

                loss_weight_cfg=config["model"]["loss_weight_cfg"],
                balance_cls_weight=config["model"]["balance_cls_weight"],
                class_frequencies=CONF.semantic_kitti_class_frequencies,
            )

            model = pl_model_lseg(
                model=model,
                model_version=model_version,
                TPV_version=TPV_version,
                config=config
            )
        
        elif TPV_version == 'v3':
            model = RefHead_TPV_Lseg_V3(
                num_class=config["model"]["num_class"],
                geo_feat_channels=32,
                img_feat_channels=2048,
                z_down=True,
                use_sgm='ce_loss',
                loss_weight_cfg=config["model"]["loss_weight_cfg"],
                balance_cls_weight=config["model"]["balance_cls_weight"],
                class_frequencies=CONF.semantic_kitti_class_frequencies,
            )

            model = pl_model_lseg(
                model=model,
                model_version=model_version,
                TPV_version=TPV_version,
                config=config
            )
        
        elif TPV_version == 'fs':
            model = RefHead_TPV_Lseg_V3_FS(
                num_class=config["model"]["num_class"],
                geo_feat_channels=32,
                z_down=True,

                loss_weight_cfg=config["model"]["loss_weight_cfg"],
                balance_cls_weight=config["model"]["balance_cls_weight"],
                class_frequencies=CONF.semantic_kitti_class_frequencies,
            )

            model = pl_model_lseg(
                model=model,
                model_version=model_version,
                TPV_version=TPV_version,
                config=config
            )


    
        dm = SemanticKITTIDataModule(
            dataset=SemanticKITTIDataset,
            data_root=CONF.PATH.DATA_ROOT,
            ann_file=CONF.PATH.DATA_LABEL,
            pred_model=config["data"]["pred_model"],
            vlm_model='Lseg'
            )

    
    elif model_version == 'text':
        ffn_cfg=dict(
            type='FFN',
            embed_dims=32,
            feedforward_channels=512,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.1,
            add_identity=True
        )

        model = RefHead_Text(
            num_class=config["model"]["num_class"],
            geo_feat_channels=32,
            ffn_cfg = ffn_cfg,

            loss_weight_cfg=config["model"]["loss_weight_cfg"],
            balance_cls_weight=config["model"]["balance_cls_weight"],
            class_frequencies=CONF.semantic_kitti_class_frequencies,
        )

        model = pl_model_text(
            model=model,
            model_version=model_version,
            config=config
            )

        
        dm = SemanticKITTIDataModule(
            dataset=SemanticKITTIDataset,
            data_root=CONF.PATH.DATA_ROOT,
            ann_file=CONF.PATH.DATA_LABEL,
            pred_model=config["data"]["pred_model"],
            text_model='Blip2',
        )


    if model_version == 'diffusion':
        model = pl_model_diff(
            model=model,
            model_version=model_version,
            config=config
        )

    elif model_version == 'lseg':
        pass

    elif model_version == 'text':
        pass

    else:
        model = pl_model(
            model=model,
            model_version=model_version,
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
        save_top_k=1,
        save_last=True,
        filename='{val/mIoU:.4f}'
        )

    # trainer
    trainer = pl.Trainer(
        devices=[i for i in range(num_gpu)],
        strategy=DDPStrategy(
            accelerator='gpu',
            find_unused_parameters=False
        ),
        max_steps=config["training"]['training_steps'],
        max_epochs=config["training"]['training_epochs'],
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

    #trainer.fit(model=model, datamodule=dm, ckpt_path="/u/home/caoh/projects/MA_Jiachen/3DPNA/output_new/ckpts/version_1/epoch_epoch=009-mIoU=val/mIoU=0.1720.ckpt")
    trainer.fit(model=model, datamodule=dm)
    
if __name__ == "__main__":
    main()

