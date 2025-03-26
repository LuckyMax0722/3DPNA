import torch
import torch.nn as nn 

import pytorch_lightning as pl

from einops import rearrange

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from configs.config import CONF

from projects.model.dfa import SemanticGuidanceModule
from projects.model.tpv.TPVAE_V2 import TPVGenerator, TPVAggregator
from projects.model.tpv.blocks import Encoder, Header, Decoder

from mmdet.models.backbones.resnet import ResNet

from projects.loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss

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


class RefHead_TPV_Lseg_V2(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,
        img_feat_channels,
        kv_dim,
        z_down,

        dim_head,
        heads,
        ffn_cfg,
    
        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
        padding_mode='replicate',
    ):
        super(RefHead_TPV_Lseg_V2, self).__init__()

        if z_down:
            shape_256 = (256, 256, 32)
            shape_128 = (128, 128, 16)
            shape_64 = (64, 64, 8)
            shape_32 = (32, 32, 4)
        else:
            raise ValueError("Wrong Shape Size.")

        self.i_embedding = nn.Embedding(num_class, num_class)  # [B, C, H, W]
        self.v_embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, C, H, W]

        self.conv_in = nn.Conv3d(
            geo_feat_channels, 
            geo_feat_channels, 
            kernel_size=(5, 5, 3), 
            stride=(1, 1, 1), 
            padding=(2, 2, 1), 
            bias=True, 
            padding_mode=padding_mode
        )

        self.geo_encoder_128 = Encoder(
            geo_feat_channels=geo_feat_channels,
            z_down=z_down, 
            padding_mode=padding_mode
        )


        self.tpv = TPVGenerator(
            embed_dims=geo_feat_channels,
            split=[8, 8, 8],
            grid_size=[128, 128, 16],
            pooler='avg',
        )

        self.fuser = TPVAggregator(
            embed_dims=geo_feat_channels,
        )

        self.sgm_128 = SemanticGuidanceModule(
            geo_feat_channels=geo_feat_channels,
            img_feat_channels=img_feat_channels // 4,
            kv_dim=kv_dim,
            shape=shape_128,
            dim_head=dim_head,
            heads=heads,
            ffn_cfg=ffn_cfg,
        )

        self.img_backbone = ResNet(
            depth=50,
            in_channels=23,
            num_stages=2,
            out_indices=(0, 1),
            strides=(1, 2),
            dilations=(1, 1),
            )

        self.decoder_256 = Decoder(
            geo_feat_channels=geo_feat_channels
        )

        self.pred_head_256 = Header(geo_feat_channels, num_class)
        self.pred_head_128 = Header(geo_feat_channels, num_class)


        # voxel losses
        self.empty_idx = empty_idx
        
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
            
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17

    def forward(self, voxel, image, image_seg):
        image_seg = self.i_embedding(image_seg)
        image_seg = rearrange(image_seg, 'b c h w emb -> b (c emb) h w')  # torch.Size([1, 1, 384, 1280, 20])

        x = voxel.detach().clone()
        x[x == 255] = 0
            
        x = self.v_embedding(x)
        x = x.permute(0, 4, 1, 2, 3)

        x_256 = self.conv_in(x)
        x_128 = self.geo_encoder_128(x_256)

        tpv_feat, _ = self.tpv(x_128)

        seg_feat = torch.cat((image_seg, image), dim=1)
        seg_feat = self.img_backbone(seg_feat)
        '''
        img_feat:
            [256, 96, 320], [512, 48, 160], [1024, 24, 80], [2048, 12, 40]
        '''

        tpv_feat_128 = self.sgm_128(tpv_feat, seg_feat[1])

        x_128, _ = self.fuser(tpv_feat_128, x_128)

        x_256 = self.decoder_256(x_256, x_128)

        x_256 = self.pred_head_256(x_256)
        x_128 = self.pred_head_128(x_128)

        return x_128, x_256

    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict

if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_tpv_lseg_v2.py
    
    from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset

    ds = SemanticKITTIDataset(
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model='CGFormer',
         vlm_model = 'Lseg',
        split='train'
        )

    image = ds[0]['img'].unsqueeze(0).cuda()
    image_seg = ds[0]['img_seg'].unsqueeze(0).cuda()
    voxel = ds[0]['input_occ'].unsqueeze(0).cuda()

    rh = RefHead_TPV_Lseg_V2(
        num_class=20,
        geo_feat_channels=32,
        img_feat_channels=2048,
        kv_dim=256,
        z_down=True,
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
        ),

        balance_cls_weight=True,
        class_frequencies=[
            5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
            8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
            4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
            1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
        ]

    ).cuda()

    rh(voxel, image, image_seg)
    
