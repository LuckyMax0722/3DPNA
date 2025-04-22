import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from cores.loss.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss, KL_loss

from cores.model.pna_modules.NeighborhoodCrossAttention import NeighborhoodCrossAttention as NCA
from cores.model.pna_modules.NeighborhoodSelfAttention import NeighborhoodSelfAttention as NSA

from cores.model.modules.FeatureExtractionBlock import FeatureExtractionBlock_Conv, FeatureExtractionBlock_PNA

from cores.model.modules.FeatureAggregationBlock import FeatureAggregationBlock_Conv, FeatureAggregationBlock_PNA

from cores.model.modules.PredHeaders import PredHeaders

from cores.model.text_modules.SemanticInteractionGuidanceModule import SemanticInteractionGuidanceModule as SIGM

class UNet(nn.Module):
    def __init__(self, 
        geo_feat_channels,
        num_class,
        text_model,
        text_dim,

        ffn_cfg,
        num_heads,
        kernel_size,
        dilation,
        rel_pos_bias,
        qkv_bias,
        attn_drop,
        proj_drop,
        use_fna
    ):
        super().__init__()
        
        self.conv0 = nn.Conv3d(
            geo_feat_channels, 
            geo_feat_channels, 
            kernel_size=(5, 5, 3), 
            stride=(1, 1, 1), 
            padding=(2, 2, 1), 
            bias=True, 
            padding_mode='replicate'
        )
        
        # Encoder Branch
        self.FEB_Conv_256 = FeatureExtractionBlock_Conv(
            geo_feat_channels
        )

        self.FEB_PNA_128 = FeatureExtractionBlock_PNA(
            geo_feat_channels=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[2],
            dilation=dilation[2],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )

        self.FEB_PNA_64 = FeatureExtractionBlock_PNA(
            geo_feat_channels=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[1],
            dilation=dilation[1],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )

        self.FEB_PNA_32 = FeatureExtractionBlock_PNA(
            geo_feat_channels=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[0],
            dilation=dilation[0],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )


        self.bottleneck = FeatureExtractionBlock_Conv(
            geo_feat_channels,
            z_down=False
        )

        # Text Branch
        self.text_model = text_model

        self.FAB_Conv_text_32 = FeatureAggregationBlock_Conv(
            geo_feat_channels, 
        )

        self.FAB_Conv_text_64 = FeatureAggregationBlock_Conv(
            geo_feat_channels, 
        )

        self.FAB_Conv_text_128 = FeatureAggregationBlock_Conv(
            geo_feat_channels, 
        )

        self.FAB_Conv_text_256 = FeatureAggregationBlock_Conv(
            geo_feat_channels, 
        )


        self.sigm_16 = SIGM(
            geo_feat_channels=geo_feat_channels,
            text_dim=text_dim,
        )

        self.sigm_32 = SIGM(
            geo_feat_channels=geo_feat_channels,
            text_dim=text_dim,
        )

        self.sigm_64 = SIGM(
            geo_feat_channels=geo_feat_channels,
            text_dim=text_dim,
        )

        self.sigm_128 = SIGM(
            geo_feat_channels=geo_feat_channels,
            text_dim=text_dim,
        )

        self.pred_head_text = PredHeaders(
            geo_feat_channels=geo_feat_channels,
            num_class=num_class
        )

    def forward(self, x, text):  # [b, geo_feat_channels, X, Y, Z]   
        
        x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
        
        # Encoder
        x, skip_256 = self.FEB_Conv_256(x) # skip1: ([1, 64, 256, 256, 32]) / x: ([1, 64, 128, 128, 16])
        x, skip_128 = self.FEB_PNA_128(x) # skip2: ([1, 64, 128, 128, 16]) / x: ([1, 64, 64, 64, 8])  
        x, skip_64 = self.FEB_PNA_64(x) # skip3: ([1, 64, 64, 64, 8]) / x: ([1, 64, 32, 32, 4])
        x, skip_32 = self.FEB_PNA_32(x) # skip4: ([1, 64, 32, 32, 4]) / x: ([1, 64, 16, 16, 2])
        
        x_16 = self.bottleneck(x) # x: ([1, 64, 16, 16, 2])
        
        x_text_16 = self.sigm_16(x_16, text)
        x_text_32 = self.FAB_Conv_text_32(x_text_16, skip_32)  # x: ([1, 64, 32, 32, 4])
        x_text_32 = self.sigm_32(x_text_32, text)
        x_text_64 = self.FAB_Conv_text_64(x_text_32, skip_64)  # x: ([1, 64, 64, 64, 8]) 
        x_text_64 = self.sigm_64(x_text_64, text)
        x_text_128 = self.FAB_Conv_text_128(x_text_64, skip_128)  # x: ([1, 64, 128, 128, 16])
        x_text_128 = self.sigm_128(x_text_128, text)
        x_text_256 = self.FAB_Conv_text_256(x_text_128, skip_256)  # x: ([1, 64, 256, 256, 32])

        # pred
        x_text_32, x_text_64, x_text_128, x_text_256 = self.pred_head_text(x_text_32, x_text_64, x_text_128, x_text_256)

        return [x_text_32, x_text_64, x_text_128, x_text_256]



class RefHead_TEXT(nn.Module):
    def __init__(
        self,
        num_class=None,
        geo_feat_channels=None,
        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
        text_model=None,
        text_dim=None,
        ffn_cfg=None,
        num_heads=None,
        kernel_size=None,
        dilation=None,
        rel_pos_bias=None,
        qkv_bias=None,
        attn_drop=None,
        proj_drop=None,
        use_fna=None,
    ):
        super(RefHead_TEXT, self).__init__()
        
        self.empty_idx = empty_idx
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        
        self.unet = UNet(
            geo_feat_channels=geo_feat_channels,
            num_class=num_class,
            text_model=text_model,
            text_dim=text_dim,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
            )

        
        # voxel losses
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
            
    def forward(self, x, text):
        x[x == 255] = 0

        x = self.embedding(x)

        x = rearrange(x, 'b h w z c -> b c h w z ')  

        output_TEXT = self.unet(x, text)

        return output_TEXT
    
    def ms_loss(self, output_voxels_list, target_voxels_list, branch):
        loss_dict = {}
        
        suffixes = [32, 64, 128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{branch}_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{branch}_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{branch}_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict


        
        
