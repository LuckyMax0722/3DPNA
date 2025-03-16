import torch
import torch.nn as nn 

import pytorch_lightning as pl


import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from configs.config import CONF
from projects.model.lseg import LSegModule
from projects.model.tpv import TPVEncoder
from projects.model.dfa import SemanticGuidanceModule

def get_image(img_path):
    crop_size = 480
    padding = [0.0] * 3
    image = Image.open(img_path)
    image = np.array(image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0)

    return image


class RefHead_TPV_Lseg(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,
        img_feat_channels,
        shape,
        dim_head,
        heads,
        ffn_cfg,
        lseg_ckpt=CONF.PATH.CKPT_LSEG,
    ):
        super(RefHead_TPV_Lseg, self).__init__()

        self.LSeg = LSegModule.load_from_checkpoint(
            checkpoint_path=lseg_ckpt, 
            backbone='clip_vitl16_384',
            num_features=256,
            arch_option=0,
            block_depth=0,
            activation='lrelu',
        ).eval()

        self.tpv = TPVEncoder(
            num_class=num_class,
            geo_feat_channels=geo_feat_channels,
            z_down=False,
            padding_mode='replicate',
        )

        self.sgm = SemanticGuidanceModule(
            geo_feat_channels=geo_feat_channels,
            img_feat_channels=img_feat_channels,
            shape=shape,
            dim_head=dim_head,
            heads=heads,
            ffn_cfg=ffn_cfg,
        )


    def forward(self, voxel, image, image_seg):
        with torch.no_grad():
            seg_feat = self.LSeg.infer(image_seg, vis=False)  # [bs, num_class, input_h, input_w]
        
        tpv_feat = self.tpv(voxel)
        '''
        [
            torch.Size([1, 32, 128, 128])
            torch.Size([1, 32, 128, 32])
            torch.Size([1, 32, 128, 32])
        ]
        '''

        self.sgm(tpv_feat, seg_feat, image)


        import subprocess

        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        # print(result.stdout)
        

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=1 python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_tpv_lseg.py
    
    from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset

    ds = SemanticKITTIDataset(
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model='CGFormer',
        split='train'
        )

    image = ds[0]['img'].unsqueeze(0).cuda()
    image_seg = ds[0]['img_seg'].unsqueeze(0).cuda()
    voxel = ds[0]['input_occ'].unsqueeze(0).cuda()

    rh = RefHead_TPV_Lseg(
        num_class=20,
        geo_feat_channels=32,
        img_feat_channels=256,
        shape=(128,128,32),
        dim_head=8,
        heads=4,
        ffn_cfg=dict(
            type='FFN',
            embed_dims=32,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.1,
            add_identity=True
        )

    ).cuda()

    rh(voxel, image, image_seg)