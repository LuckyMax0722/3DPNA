import torch
import torch.nn as nn
import torch.nn.functional as F

from cores.model.pna_modules.natten_utils.NeighborhoodCrossAttentionModule import NeighborhoodCrossAttentionModule as NCAM
from cores.model.pna_modules.natten_utils.SelfAttentionModule import SelfAttentionModule as SAM

from mmcv.cnn.bricks.transformer import build_feedforward_network

from einops import rearrange

class ProgressiveNeighborhoodAggregation(nn.Module): # progressive neighborhood aggregation
    def __init__(
        self,
        geo_feat_channels=None,
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

        super(ProgressiveNeighborhoodAggregation, self).__init__()

        embed_dims = geo_feat_channels

        self.sam = SAM(
            embed_dims=embed_dims,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
        )

        self.cam = NCAM(
            dim=embed_dims, 
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=True,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
            )
        
        # FFN Layer
        self.ffn = build_feedforward_network(ffn_cfg)

        # Upscale x
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Norm Layer
        self.norm_cam_x =nn.InstanceNorm3d(embed_dims)
        self.norm_cam_v =nn.InstanceNorm3d(embed_dims)

        # Norm Layer
        self.norm =nn.InstanceNorm3d(embed_dims)
        self.norm_out =nn.InstanceNorm3d(embed_dims)

    def forward(self, skip, x):
        """
        input:
            x: torch.size([1, 64, 16, 16, 2])/....
            skip: torch.size([1, 64, 32, 32, 4])/....
        
        x: (B, C, H, W, Z)
        """

        b, c, h, w, z = skip.shape

        feat_s = self.sam(skip)

        # Upscale to skip dim
        x = self.up_scale(x)  

        # Input Norm
        x = self.norm_cam_x(x)
        skip = self.norm_cam_v(skip)

        feat_c = self.cam(x, skip)

        # Output Norm
        out = self.norm(feat_c + feat_s)

        # FFN Rearrange
        out = rearrange(out, 'b c h w z -> b (h w z) c')

        # FFN FeadForward
        out = self.ffn(out)

        # FFN output
        out = rearrange(out, 'b (h w z) c -> b c h w z', h=h, w=w, z=z)

        # Model Output
        out = self.norm_out(out)

        return out

