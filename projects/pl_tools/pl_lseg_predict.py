import os
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from configs.config import CONF
from projects.model.lseg import LSegModule

from projects.datasets import SemanticKITTILsegDataModule, SemanticKITTILsegDataset

def main():
    num_gpu = torch.cuda.device_count()

    dm = SemanticKITTILsegDataModule(
        dataset=SemanticKITTILsegDataset,
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model='CGFormer',
        )

    model = LSegModule.load_from_checkpoint(
        checkpoint_path=CONF.PATH.CKPT_LSEG, 
        backbone='clip_vitl16_384',
        num_features=256,
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )
    
    trainer = pl.Trainer(
        devices=[i for i in range(num_gpu)],
        strategy=DDPStrategy(
            accelerator='gpu',
            find_unused_parameters=False
        ),
    )

    trainer.predict(model=model, datamodule=dm)


if __name__ == '__main__':
    main()

    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/pl_tools/pl_lseg_predict.py