import os
import yaml

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor


from configs.config import CONF
from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_config(baseline_model):
    yaml_config = os.path.join(CONF.PATH.CONFIG_DIR, ('REF_' + baseline_model + '.yaml'))

    with open(yaml_config, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def get_logger(debug):
    if debug:

        logger = False

        callbacks=[]

    else:

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
            save_top_k=1,
            save_last=True,
            filename='{val/mIoU:.4f}'
            )
        
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step'),
        ]

    return logger, callbacks

def get_model(model_version, text_model, config):
    ffn_cfg=dict(
            type='FFN',
            embed_dims=32,
            feedforward_channels=1024,
            num_fcs=3,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.1,
            add_identity=True
        )

    if model_version == 'pna_v3':
        from projects.model.rh_pna_v3 import RefHead_PNA_V3
        from projects.pl_tools.pl_model_pna import pl_model_pna

        kernel_size = [[3,3,3], [3,3,3], [5,5,5]]
        dilation = [[1,1,1], [1,1,1], [1,1,1]]

        model = RefHead_PNA_V3(
            num_class=config["model"]["num_class"],
            geo_feat_channels=32,

            loss_weight_cfg=config["model"]["loss_weight_cfg"],
            balance_cls_weight=config["model"]["balance_cls_weight"],
            class_frequencies=CONF.semantic_kitti_class_frequencies,

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

        model = pl_model_pna(
            model=model,
            model_version=model_version,
            config=config
            )

        
        dm = SemanticKITTIDataModule(
            dataset=SemanticKITTIDataset,
            data_root=CONF.PATH.DATA_ROOT,
            ann_file=CONF.PATH.DATA_LABEL,
            pred_model=config["data"]["pred_model"],
            text_model=text_model,
        )

    return model, dm

def main():
    model_version = 'pna_v3'  # small, pna, vqvae cvae, diffusion, lseg, pna_v2, text
    baseline_model = 'CGFormer'  # CGFormer, MonoScene
    text_model = None # BLIP2, CLIP, LongCLIP, JinaCLIP, JinaCLIP_1024

    ckpt_path = None
    debug = False

    config = get_config(baseline_model)

    seed = config["training"]["seed"]
    pl.seed_everything(seed)

    # Get Logger, Callbacks
    logger, callbacks = get_logger(debug)

    # Get Model, DataModule
    model, dm = get_model(model_version, text_model, config)

    # trainer
    trainer = pl.Trainer(
        devices=[i for i in range(torch.cuda.device_count())],
        strategy=DDPStrategy(
            accelerator='gpu',
            find_unused_parameters=False
        ),
        max_steps=config["training"]['training_steps'],
        max_epochs=config["training"]['training_epochs'],
        callbacks=callbacks,
        logger=logger,
        profiler="simple",
        sync_batchnorm=True,
        log_every_n_steps=config["training"]['log_every_n_steps'],
        check_val_every_n_epoch=config["training"]['check_val_every_n_epoch']
    )

    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)
    
if __name__ == "__main__":
    main()

# python /u/home/caoh/projects/MA_Jiachen/3DPNA/tools/train_pna.py