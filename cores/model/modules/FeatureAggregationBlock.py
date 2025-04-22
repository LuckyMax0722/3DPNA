import torch
import torch.nn as nn
import torch.nn.functional as F

from cores.model.pna_modules.NeighborhoodCrossAttention import NeighborhoodCrossAttention as NCA
from cores.model.pna_modules.NeighborhoodCrossAttention import NeighborhoodCrossAttentionV2 as NCAV2
from cores.model.pna_modules.NeighborhoodCrossAttention import NeighborhoodCrossAttentionV3 as NCAV3
from cores.model.modules.Base import ResConvBlock

class FeatureAggregationBlock_Conv(nn.Module):
    def __init__(
        self, 
        geo_feat_channels
    ):

        super().__init__()
        
        self.convblock = ResConvBlock(geo_feat_channels=geo_feat_channels)
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, x, skip):
        
        x = self.up_scale(x)
        x = self.convblock(skip, x)
        
        return x


class FeatureAggregationBlock_PNA(nn.Module): # FeatureAggregationBlock_PNA
    def __init__(
        self,
        geo_feat_channels,
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
        super(FeatureAggregationBlock_PNA, self).__init__()

        self.pna_decoder = NCA(
            embed_dims=geo_feat_channels,
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

    def forward(self, skip, x):
        """
        input:
            x: torch.size([1, 64, 16, 16, 2])/....
            skip: torch.size([1, 64, 32, 32, 4])/....

        x: (B, C, H, W, Z)
        """
        
        x = self.pna_decoder(skip, x)

        return x
    

class FeatureAggregationBlock_PNAV2(nn.Module): # FeatureAggregationBlock_PNA
    def __init__(
        self,
        geo_feat_channels,
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
        super(FeatureAggregationBlock_PNAV2, self).__init__()

        self.pna_decoder = NCAV2(
            embed_dims=geo_feat_channels,
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

    def forward(self, skip, x):
        """
        input:
            x: torch.size([1, 64, 16, 16, 2])/....
            skip: torch.size([1, 64, 32, 32, 4])/....

        x: (B, C, H, W, Z)
        """
        
        x = self.pna_decoder(skip, x)

        return x

class FeatureAggregationBlock_PNAV3(nn.Module): # FeatureAggregationBlock_PNA
    def __init__(
        self,
        geo_feat_channels,
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
        super(FeatureAggregationBlock_PNAV3, self).__init__()

        self.pna_decoder = NCAV3(
            embed_dims=geo_feat_channels,
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

    def forward(self, skip, x):
        """
        input:
            x: torch.size([1, 64, 16, 16, 2])/....
            skip: torch.size([1, 64, 32, 32, 4])/....

        x: (B, C, H, W, Z)
        """
        
        x = self.pna_decoder(skip, x)

        return x