import torch
import torch.nn as nn
import torch.nn.functional as F

from cores.model.pna_modules.SelfAttentionModule import SelfAttentionModule as SAM

from mmcv.cnn.bricks.transformer import build_feedforward_network

from einops import rearrange

class NeighborhoodSelfAttention(nn.Module): # NeighborhoodSelfAttention
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
        super(NeighborhoodSelfAttention, self).__init__()

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

        # FFN Layer
        self.ffn = build_feedforward_network(ffn_cfg)

        # Norm Layer
        self.norm_in =nn.InstanceNorm3d(embed_dims)
        self.norm_out =nn.InstanceNorm3d(embed_dims)

    def forward(self, x):
        """
        input:
            x: torch.size([1, 64, 16, 16, 2])/....

        x: (B, C, H, W, Z)
        """
        b, c, h, w, z = x.shape

        x = self.sam(x)

        # Output Norm
        x = self.norm_in(x)

        # FFN Rearrange
        x = rearrange(x, 'b c h w z -> b (h w z) c')

        # FFN FeadForward
        x = self.ffn(x)

        # FFN output
        x = rearrange(x, 'b (h w z) c -> b c h w z', h=h, w=w, z=z)

        # Model Output
        x = self.norm_out(x)

        return x

