import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from mmcv.cnn.bricks.transformer import build_feedforward_network

class SelfAttentionModule(nn.Module): # SelfAttentionModule
    def __init__(
        self,
        embed_dims=None,
        use_residual=True,
        ffn_cfg=None,
        num_heads=None,
        bias=False,
    ):
        super(SelfAttentionModule, self).__init__()

        self.embed_dims = embed_dims
        self.use_residual = use_residual
        self.num_heads = num_heads

        # FFN Layer
        self.ffn = build_feedforward_network(ffn_cfg)

        # Norm Later
        self.norm1 =nn.InstanceNorm3d(embed_dims)
        self.norm2 =nn.InstanceNorm3d(embed_dims)

        # Generate QKV
        self.qkv = nn.Conv3d(embed_dims, embed_dims * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv3d(embed_dims * 3, embed_dims * 3, kernel_size=3, stride=1, padding=1, groups=embed_dims * 3, bias=bias)

        # temperature
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # output
        self.project_out = nn.Conv3d(embed_dims, embed_dims, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W, Z)
        return: (B, C, H, W, Z)
        """
        b, c, h, w, z = x.shape

        identity = x if self.use_residual else None

        # Norm
        x = self.norm1(x)

        # Generate QKV
        qkv = self.qkv_dwconv(self.qkv(x))  # shape: (B, 3*C, X, Y, Z)
        q, k, v = qkv.chunk(3, dim=1)  # q,k,v shape: (B, C, X, Y, Z)

        # Multiple Head --> torch.Size([1, 8, 8, h*w*z])
        q = rearrange(q, 'b (head c1) h w z -> b head c1 (h w z)', head=self.num_heads)
        k = rearrange(k, 'b (head c1) h w z -> b head c1 (h w z)', head=self.num_heads)
        v = rearrange(v, 'b (head c1) h w z -> b head c1 (h w z)', head=self.num_heads)

        # Norm
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Attention  Q*K
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # Attention  *V
        out = torch.matmul(attn, v)  # torch.Size([1, 8, 8, 2097152])

        # Reshape
        out = rearrange(
            out,
            'b head c1 (h w z) -> b (head c1) h w z',
            head=self.num_heads, h=h, w=w, z=z
        )

        # output
        out = self.project_out(out)  # (B, C, D, H, W)

        # Norm
        out = self.norm2(identity + out)

        # FFN Rearrange
        out = rearrange(out, 'b c h w z -> b (h w z) c')
        identity = rearrange(identity, 'b c h w z -> b (h w z) c')

        # FFN FeadForward
        out = self.ffn(out, identity)

        # FFN output
        out = rearrange(out, 'b (h w z) c -> b c h w z', h=h, w=w, z=z)

        return out