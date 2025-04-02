import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange

import numpy as np


from configs.config import CONF

from projects.model.tpv.TPVAE_V2 import TPVGenerator, TPVAggregator
from projects.model.tpv.blocks import Encoder, Header, Decoder

from projects.loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss

class RefHead_TPV_Lseg_V3_FS(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,
        z_down,
        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
        padding_mode='replicate',
    ):
        super(RefHead_TPV_Lseg_V3_FS, self).__init__()

        if z_down:
            shape_256 = [256, 256, 32]
            shape_128 = [128, 128, 16]
            shape_64 = [64, 64, 8]
            shape_32 = [32, 32, 4]
        else:
            raise ValueError("Wrong Shape Size.")

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

        self.geo_encoder_64 = Encoder(
            geo_feat_channels=geo_feat_channels,
            z_down=z_down, 
            padding_mode=padding_mode
        )

        self.geo_encoder_32 = Encoder(
            geo_feat_channels=geo_feat_channels,
            z_down=z_down, 
            padding_mode=padding_mode
        )


        self.tpv_256 = TPVGenerator(
            embed_dims=geo_feat_channels,
            split=[8, 8, 8],
            grid_size=shape_256,
            pooler='avg',
        )

        self.tpv_128 = TPVGenerator(
            embed_dims=geo_feat_channels,
            split=[8, 8, 8],
            grid_size=shape_128,
            pooler='avg',
        )

        self.tpv_64 = TPVGenerator(
            embed_dims=geo_feat_channels,
            split=[8, 8, 8],
            grid_size=shape_64,
            pooler='avg',
        )

        self.tpv_32 = TPVGenerator(
            embed_dims=geo_feat_channels,
            split=[8, 8, 8],
            grid_size=shape_32,
            pooler='avg',
        )

        self.fuser_256 = TPVAggregator(
            embed_dims=geo_feat_channels,
        )

        self.fuser_128 = TPVAggregator(
            embed_dims=geo_feat_channels,
        )

        self.fuser_64 = TPVAggregator(
            embed_dims=geo_feat_channels,
        )

        self.fuser_32 = TPVAggregator(
            embed_dims=geo_feat_channels,
        )

        self.decoder_256 = Decoder(
            geo_feat_channels=geo_feat_channels
        )

        self.decoder_128 = Decoder(
            geo_feat_channels=geo_feat_channels
        )

        self.decoder_64 = Decoder(
            geo_feat_channels=geo_feat_channels
        )


        self.pred_head_256 = Header(geo_feat_channels, num_class)
        self.pred_head_128 = Header(geo_feat_channels, num_class)
        self.pred_head_64 = Header(geo_feat_channels, num_class)
        self.pred_head_32 = Header(geo_feat_channels, num_class)


        self.sh_256 = SemanticHead(
            in_channels=geo_feat_channels, 
            num_classes=num_class,
            )
        
        self.sh_128 = SemanticHead(
            in_channels=geo_feat_channels, 
            num_classes=num_class,
            )

        self.sh_64 = SemanticHead(
            in_channels=geo_feat_channels, 
            num_classes=num_class,
            )

        self.sh_32 = SemanticHead(
            in_channels=geo_feat_channels, 
            num_classes=num_class,
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

    def forward(self, voxel, image_seg):
        x = voxel.detach().clone()
        x[x == 255] = 0
            
        x = self.v_embedding(x)
        x = x.permute(0, 4, 1, 2, 3)

        x_256 = self.conv_in(x)
        x_128 = self.geo_encoder_128(x_256)
        x_64 = self.geo_encoder_64(x_128)
        x_32 = self.geo_encoder_32(x_64)

        tpv_feat_256, _ = self.tpv_256(x_256)
        tpv_feat_128, _ = self.tpv_128(x_128)
        tpv_feat_64, _ = self.tpv_64(x_64)
        tpv_feat_32, _ = self.tpv_32(x_32)

        if self.training:
            loss_dict = {}

            loss_dict[f'loss_ce_sh_256'] = self.sh_256(x = tpv_feat_256[2], s = image_seg)
            loss_dict[f'loss_ce_sh_128'] = self.sh_128(x = tpv_feat_128[2], s = image_seg)
            loss_dict[f'loss_ce_sh_64'] = self.sh_64(x = tpv_feat_64[2], s = image_seg)
            loss_dict[f'loss_ce_sh_32'] = self.sh_32(x = tpv_feat_32[2], s = image_seg)

        else:
            loss_dict = {}
            
        
        tpv_feat_256 = self.tpv_unsqueeze(tpv_feat_256)
        tpv_feat_128 = self.tpv_unsqueeze(tpv_feat_128)
        tpv_feat_64 = self.tpv_unsqueeze(tpv_feat_64)
        tpv_feat_32 = self.tpv_unsqueeze(tpv_feat_32)
        
        x_256, _ = self.fuser_256(tpv_feat_256, x_256)
        x_128, _ = self.fuser_128(tpv_feat_128, x_128)
        x_64, _ = self.fuser_64(tpv_feat_64, x_64)
        x_32, _ = self.fuser_32(tpv_feat_32, x_32)


        x_256 = self.decoder_256(x_256, x_128)
        x_128 = self.decoder_128(x_128, x_64)
        x_64 = self.decoder_64(x_64, x_32)

        x_256 = self.pred_head_256(x_256)
        x_128 = self.pred_head_128(x_128)
        x_64 = self.pred_head_64(x_64)
        x_32 = self.pred_head_32(x_32)

        return x_32, x_64, x_128, x_256, loss_dict

    def tpv_unsqueeze(self, tpv_feat):

        tpv_feat[0] = tpv_feat[0].unsqueeze(-1)
        tpv_feat[1] = tpv_feat[1].unsqueeze(2)
        tpv_feat[2] = tpv_feat[2].unsqueeze(3)

        return tpv_feat

    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [32, 64, 128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict


class SemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemanticHead, self).__init__()

        self.semantic_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def semantic_gt_resize(self, x, s):
        # ----------------------------------------------------
        # 1) 下采样伪语义分割掩码 S 使其与 X 的 spatial 分辨率对齐
        # ----------------------------------------------------
        # X.shape = [1, C, Y, Z] = [1, 32, 256, 32]
        # S.shape = [1, 1, 384, 1280]
        # 我们用最近邻插值 (nearest) 的方式下采样到 (H=32, W=256)

        x = rearrange(x, 'b c y z -> b c z y')
        # X.shape = [1, C, H, W] = [1, 32, 32, 256]

        B, Cx, H, W = x.shape  # B=1, Cx=32, H=32, W=256
        s = F.interpolate(
            s.float(), 
            size=(H, W), 
            mode='nearest'
        ).long()  # 变为 [1, 1, 32, 256]

        s = s.squeeze(1) # 变为 [1, 32, 256]

        return x, s

    def forward(self, x, s):
        x, s = self.semantic_gt_resize(x, s)
        x = self.semantic_head(x)

        loss = F.cross_entropy(x, s)

        return loss


if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_tpv_lseg_v3_fs.py
    
    from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset

    ds = SemanticKITTIDataset(
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model='CGFormer',
        vlm_model = 'Lseg',
        split='train'
        )

    image_seg = ds[0]['img_seg'].unsqueeze(0).cuda()
    voxel = ds[0]['input_occ'].unsqueeze(0).cuda()

    rh = RefHead_TPV_Lseg_V3_FS(
        num_class=20,
        geo_feat_channels=32,
        z_down=True,

        balance_cls_weight=True,
        class_frequencies=[
            5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
            8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
            4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
            1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
        ]

    ).cuda()

    tpv, loss_dict = rh(voxel, image_seg)

    for i in range(4):
        print(tpv[i].size())
    
    # print(voxel.size())
    # print(image_seg.size())

    # yz = torch.randn(1, 32, 256, 32).cuda()
    # print(yz.size())

    # contrastive_loss_with_pseudo_masks(
    #     X = yz, 
    #     S = image_seg
    # )

