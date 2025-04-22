import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmcv.cnn.bricks.transformer import build_feedforward_network
from projects.model.pna_modules.natten_utils import NeighborhoodSelfAttention3D as NSA3D

class SelfAggregationModuleV2(nn.Module):
    def __init__(
        self,
        embed_dims=None,
        num_heads=8,
        kernel_size=[3,3,3],
        dilation=[1,1,1],
        rel_pos_bias=True,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_fna=False,
    ):
        super(SelfAggregationModuleV2, self).__init__()

        # Norm Later
        self.norm1 =nn.InstanceNorm3d(embed_dims)

        # NeighborhoodSelfAttention3D Layer
        self.nsa = NSA3D(
            dim = embed_dims,
            num_heads = num_heads,
            kernel_size = kernel_size,
            dilation = dilation,
            rel_pos_bias = rel_pos_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna = use_fna
        )

    def forward(self, x):
        """
        x: (B, C, H, W, Z)
        return: (B, C, H, W, Z)
        """

        # Input Norm
        x = self.norm1(x)

        out = self.nsa(x)

        return out


if __name__ == "__main__":
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/pna_modules/SelfAggregation_V2.py
    SAM = SelfAggregationModuleV2(
        embed_dims=32, 
        num_heads=8,
        kernel_size=[3,3,3],
        dilation=[1,1,1],
        rel_pos_bias=True,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        use_fna=False,
    )

    tensor_shape = (1, 32, 32, 32, 4)
    x = torch.randn(tensor_shape)

    y = SAM(x)
    print(y.size())