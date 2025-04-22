import torch
import torch.nn as nn

from cores.model.pna_modules.NeighborhoodSelfAttention import NeighborhoodSelfAttention as NSA

from cores.model.pna_modules.natten_utils.NeighborhoodSelfAttentionModuleV2 import NeighborhoodSelfAttentionModuleV2 as NSAM

from cores.model.modules.Base import ConvBlock

class FeatureExtractionBlock_Conv(nn.Module):
    def __init__(
        self, 
        geo_feat_channels, 
        z_down=True
    ):

        super().__init__()
        self.z_down = z_down

        self.convblock = ConvBlock(input_channels=geo_feat_channels, output_channels=geo_feat_channels)

        if z_down :
            self.downsample = nn.MaxPool3d((2, 2, 2))

    
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        residual_feat = x
        x = self.convblock(x)  # [b, geo_feat_channels, X, Y, Z]
        skip = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
    
        if self.z_down:
            x = self.downsample(skip)  # [b, geo_feat_channels, X//2, Y//2, Z//2]

            return x, skip
        else:
            return skip  # [b, geo_feat_channels, X, Y, Z]


class FeatureExtractionBlock_PNA(nn.Module): # FeatureExtractionBlock_PNA
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
        super(FeatureExtractionBlock_PNA, self).__init__()

        self.pna_encoder = NSA(
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

        # Downsample
        self.downsample = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        """
        input:
            x: torch.size([1, 64, 16, 16, 2])/....

        x: (B, C, H, W, Z)
        """
        residual_feat = x

        x = self.pna_encoder(x)

        skip = x + residual_feat

        x = self.downsample(skip)

        return x, skip


class FeatureExtractionBlock_PNAV2(nn.Module): # FeatureExtractionBlock_PNA
    def __init__(
        self,
        geo_feat_channels,
        ffn_cfg,
        num_heads,
    ):
        super(FeatureExtractionBlock_PNAV2, self).__init__()

        self.pna_encoder = NSAM(
            embed_dims=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
        )

        # Downsample
        self.downsample = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        """
        input:
            x: torch.size([1, 64, 16, 16, 2])/....

        x: (B, C, H, W, Z)
        """
        residual_feat = x

        x = self.pna_encoder(x)

        skip = x + residual_feat

        x = self.downsample(skip)

        return x, skip

