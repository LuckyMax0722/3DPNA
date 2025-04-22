#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_

import natten
from natten.functional import na3d, na3d_av, na3d_qk
from natten.types import CausalArg3DTypeOrDed, Dimension3DTypeOrDed
from natten.utils import check_all_args, log

logger = log.get_logger(__name__)

from einops import rearrange

class NeighborhoodSelfAttentionModule(nn.Module):
    """
    Neighborhood Self Attention 3D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: Dimension3DTypeOrDed,
        dilation: Dimension3DTypeOrDed = 1,
        is_causal: CausalArg3DTypeOrDed = False,
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_fna = False
    ):
        super().__init__()
        kernel_size_, dilation_, is_causal_ = check_all_args(
            3, kernel_size, dilation, is_causal
        )
        assert len(kernel_size_) == len(dilation_) == len(is_causal_) == 3
        if any(is_causal_) and rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size_
        self.dilation = dilation_
        self.is_causal = is_causal_
        self.use_fna = use_fna

        # Generate QKV
        self.embed_qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * self.kernel_size[0] - 1),
                    (2 * self.kernel_size[1] - 1),
                    (2 * self.kernel_size[2] - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)

        # Out Proj
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-5 input tensor; got {x.dim()=}."
            )

        b, c, h, w, z = x.shape

        # Generate QKV
        qkv = self.embed_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)  # q,k shape: (B, C, X, Y, Z)

        if self.use_fna:
            natten.use_fused_na(True, kv_parallel=False)

            if self.attn_drop_rate > 0:
                logger.error(
                    "You're using fused neighborhood attention, and passed in a "
                    "non-zero attention dropout rate. This implementation does "
                    "support attention dropout yet, which means dropout is NOT being applied "
                    "to your attention weights."
                )
            
            # Multiple Head
            q = rearrange(q, 'b (head c1) h w z -> b h w z head c1', head=self.num_heads)  
            k = rearrange(k, 'b (head c1) h w z -> b h w z head c1', head=self.num_heads)
            v = rearrange(v, 'b (head c1) h w z -> b h w z head c1', head=self.num_heads)
            
            x = na3d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )

            x = rearrange(x, 'b h w z head c1 -> b (head c1) h w z')

        else:
            # Multiple Head
            q = rearrange(q, 'b (head c1) h w z -> b head h w z c1', head=self.num_heads)  
            k = rearrange(k, 'b (head c1) h w z -> b head h w z c1', head=self.num_heads)
            v = rearrange(v, 'b (head c1) h w z -> b head h w z c1', head=self.num_heads)

            q = q * self.scale

            attn = na3d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = na3d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            x = rearrange(x, 'b head h w z c1 -> b (head c1) h w z')
        
        return self.proj_drop(self.proj(x))