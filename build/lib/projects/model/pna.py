import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.model.pna_modules import SelfAggregationModule as SAM
from projects.model.pna_modules import CrossAggregationModule as CAM
from mmcv.cnn.bricks.transformer import build_feedforward_network

from einops import rearrange

class ProgressiveNeighborhoodAggregation(nn.Module): # progressive neighborhood aggregation
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
        super(ProgressiveNeighborhoodAggregation, self).__init__()

        self.sam = SAM(
            embed_dims=embed_dims,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
        )

        self.cam = CAM(
            embed_dims=embed_dims, 
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=True,
            qkv_bias=qkv_bias,
            attn_drop=0.0,
            proj_drop=0.0,
            use_fna=use_fna,
        )

        # FFN Layer
        self.ffn = build_feedforward_network(ffn_cfg)

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

if __name__ == "__main__":
    ffn_cfg=dict(
        type='FFN',
        embed_dims=64,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type='ReLU', inplace=True),
        ffn_drop=0.1,
        add_identity=True
    )

    PNA = ProgressiveNeighborhoodAggregation(
        embed_dims=64, 
        num_heads=8,
        ffn_cfg = ffn_cfg,
        kernel_size=[5,5,5]
    )

    skip = (1, 64, 64, 64, 8)
    x = (1, 64, 32, 32, 4)

    skip = torch.randn(skip)
    x = torch.randn(x)

    y = PNA(x, skip)

    print(y.size())