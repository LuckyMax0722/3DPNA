import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.model.pna_modules.SelfAggregation_V2 import SelfAggregationModuleV2 as SAM
from projects.model.pna_modules.CrossAggregation_V2 import CrossAggregationModuleV2 as CAM

from mmcv.cnn.bricks.transformer import build_feedforward_network

from einops import rearrange

class NeighborhoodAggregationDecoder(nn.Module): # progressive neighborhood aggregation
    def __init__(
        self,
        embed_dims=None,
        ffn_cfg=None,
        num_heads=8,
        kernel_size=[3,3,3],
        dilation=[1,1,1],
        rel_pos_bias=True,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        use_fna=False,
    ):
        super(NeighborhoodAggregationDecoder, self).__init__()

        self.sam = SAM(
            embed_dims=embed_dims, 
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=attn_drop,
            use_fna=use_fna,
        )

        self.cam = CAM(
            embed_dims=embed_dims, 
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=attn_drop,
            use_fna=use_fna,
        )

        # FFN Layer
        self.ffn = build_feedforward_network(ffn_cfg)

        # Norm Layer
        self.norm_in =nn.InstanceNorm3d(embed_dims)
        self.norm_out =nn.InstanceNorm3d(embed_dims)

    def forward(self, x, skip):
        """
        input:
            x: torch.size([1, 64, 16, 16, 2])/....
            skip: torch.size([1, 64, 32, 32, 4])/....

        x: (B, C, H, W, Z)
        """

        b, c, h, w, z = skip.shape

        feat_s = self.sam(skip)

        feat_c = self.cam(x, skip)

        # Output Norm
        out = self.norm_in(feat_c + feat_s)

        # FFN Rearrange
        out = rearrange(out, 'b c h w z -> b (h w z) c')

        # FFN FeadForward
        out = self.ffn(out)

        # FFN output
        out = rearrange(out, 'b (h w z) c -> b c h w z', h=h, w=w, z=z)

        # Model Output
        out = self.norm_out(out)

        return out
