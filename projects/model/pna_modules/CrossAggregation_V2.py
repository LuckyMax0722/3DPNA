import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmcv.cnn.bricks.transformer import build_feedforward_network
from projects.model.pna_modules.natten_utils import NeighborhoodCrossAttention3D as NCA3D

class CrossAggregationModuleV2(nn.Module):
    def __init__(
        self,
        embed_dims=64,
        num_heads=8,
        kernel_size=[3,3,3],
        dilation=[1,1,1],
        rel_pos_bias=True,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_fna=False,
    ):
        super(CrossAggregationModuleV2, self).__init__()

        # Upscale x
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Norm Layer
        self.norm1 =nn.InstanceNorm3d(embed_dims)
        self.norm2 =nn.InstanceNorm3d(embed_dims)

        # NeighborhoodCrossAttention3D Layer
        self.nca = NCA3D(
            dim = embed_dims,
            num_heads = num_heads,
            kernel_size = kernel_size,
            dilation = dilation,
            rel_pos_bias = rel_pos_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna = use_fna
        )

    def forward(self, x, skip):
        '''
        input:
            x: torch.size([1, 64, 16, 16, 2])/....
            v <-- skip: torch.size([1, 64, 32, 32, 4])/....
        
        x: (B, C, H, W, Z)
        '''

        # Upscale to skip dim
        x = self.up_scale(x)  

        # Input Norm
        x = self.norm1(x)
        skip = self.norm2(skip)

        out = self.nca(skip, x)

        return out
       


if __name__ == "__main__":
    CAM = CrossAggregationModule(
        embed_dims=64,
        num_heads=8,
        kernel_size=[3,3,3],
        dilation=[2,2,2],
        rel_pos_bias=True,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        use_fna=True,
    ).cuda()

    v = (1, 64, 256, 256, 32)
    qk = (1, 64, 128, 128, 16)
    qk = torch.randn(qk).cuda()
    v = torch.randn(v).cuda()

    y = CAM(qk, v)

    print(y.size())
