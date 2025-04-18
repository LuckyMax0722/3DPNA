import torch
import torch.nn as nn 

import pytorch_lightning as pl

from einops import rearrange

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from configs.config import CONF
from projects.model.lseg import LSegModule
from projects.model.tpv import TPVAE
from projects.model.dfa import SemanticGuidanceModule

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


class RefHead_TPV_Lseg(nn.Module):
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
    ):
        super(RefHead_TPV_Lseg, self).__init__()

        if z_down:
            shape_256 = (256, 256, 32)
            shape_128 = (128, 128, 16)
            shape_64 = (64, 64, 8)
            shape_32 = (32, 32, 4)
        else:
            raise ValueError("Wrong Shape Size.")

        self.embedding = nn.Embedding(num_class, num_class)  # [B, C, H, W]

        self.tpv = TPVAE(
            num_class=num_class,
            geo_feat_channels=geo_feat_channels,
            z_down=z_down,
            padding_mode='replicate',
        )

        self.sgm_32 = SemanticGuidanceModule(
            geo_feat_channels=geo_feat_channels,
            img_feat_channels=img_feat_channels,
            kv_dim=kv_dim,
            shape=shape_32,
            dim_head=dim_head,
            heads=heads,
            ffn_cfg=ffn_cfg,
        )

        self.sgm_64 = SemanticGuidanceModule(
            geo_feat_channels=geo_feat_channels,
            img_feat_channels=img_feat_channels // 2,
            kv_dim=kv_dim,
            shape=shape_64,
            dim_head=dim_head,
            heads=heads,
            ffn_cfg=ffn_cfg,
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
            #num_stages=1,
            #out_indices=(0,),
            #strides=(1,),
            #dilations=(1,),
            )

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
        image_seg = self.embedding(image_seg)

        image_seg = rearrange(image_seg, 'b c h w emb -> b (c emb) h w')  # torch.Size([1, 1, 384, 1280, 20])

        tpv_feat, vol_feat_map = self.tpv.encoder(voxel)
        '''
        Output:
            tpv_feat: [
                torch.Size([1, 32, 32, 32])
                torch.Size([1, 32, 32, 4])
                torch.Size([1, 32, 32, 4])
            ]
            vol_feat_map: [
                torch.Size([1, 32, 256, 256, 32])
                torch.Size([1, 32, 128, 128, 16])
                torch.Size([1, 32, 64, 64, 8])
                torch.Size([1, 32, 32, 32, 4])
            ]
        '''

        seg_feat = torch.cat((image_seg, image), dim=1)
        seg_feat = self.img_backbone(seg_feat)
        '''
        img_feat:
            [256, 96, 320], [512, 48, 160], [1024, 24, 80], [2048, 12, 40]
        '''

        tpv_feat_32 = self.sgm_32(tpv_feat[3], seg_feat[3])
        tpv_feat_64 = self.sgm_64(tpv_feat[2], seg_feat[2])
        tpv_feat_128 = self.sgm_128(tpv_feat[1], seg_feat[1])
        #tpv_feat_256 = self.sgm_256(tpv_feat[0], seg_feat[0])
            
        tpv_feat = [[], tpv_feat_128, tpv_feat_64, tpv_feat_32]
        x_32, x_64, x_128, x_256 = self.tpv.decoder(tpv_feat, vol_feat_map)

        return x_32, x_64, x_128, x_256

    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [32, 64, 128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=1 python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_tpv_lseg.py
    
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

    rh = RefHead_TPV_Lseg(
        num_class=20,
        geo_feat_channels=32,
        img_feat_channels=2048,
        kv_dim=256,
        z_down=True,
        dim_head=16,
        heads=2,
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