import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmcv.cnn.bricks.transformer import build_feedforward_network
from cores.model.pna_modules.natten_utils.NeighborhoodCrossAttentionModule import NeighborhoodCrossAttentionModule as NCAM

class CrossAttentionModule(nn.Module):
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
        super(CrossAttentionModule, self).__init__()

        # Norm Layer
        self.norm1 =nn.InstanceNorm3d(embed_dims)
        self.norm2 =nn.InstanceNorm3d(embed_dims)

        # Multi-Head Neighborhood Cross Attention Module (NCAM)
        self.ncam = NCAM(
            dim = embed_dims,
            num_heads = num_heads,
            kernel_size = kernel_size,
            dilation = dilation,
            rel_pos_bias = rel_pos_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna = use_fna
        )

    def forward(self, skip, up):
        '''
        input:
            qk <-- skip: torch.size([1, 64, 32, 32, 4])/....
            v <-- up: torch.size([1, 64, 32, 32, 4])/....
        '''

        # Input Norm
        up = self.norm1(up)
        skip = self.norm2(skip)

        out = self.ncam(skip, up)

        return out
    


class CrossAttentionModuleV2(nn.Module):
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
        super(CrossAttentionModuleV2, self).__init__()

        # Norm Layer
        self.norm1 =nn.InstanceNorm3d(embed_dims)
        self.norm2 =nn.InstanceNorm3d(embed_dims)

        # Multi-Head Neighborhood Cross Attention Module (NCAM)
        self.ncam = NCAM(
            dim = embed_dims,
            num_heads = num_heads,
            kernel_size = kernel_size,
            dilation = dilation,
            rel_pos_bias = rel_pos_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna = use_fna
        )

    def forward(self, skip, up):
        '''
        input:
            qk <-- skip: torch.size([1, 64, 32, 32, 4])/....
            v <-- up: torch.size([1, 64, 32, 32, 4])/....
        '''

        # Input Norm
        up = self.norm1(up)
        skip = self.norm2(skip)

        out = self.ncam(up, skip)

        return out