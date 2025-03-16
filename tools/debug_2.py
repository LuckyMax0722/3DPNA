import torch
import torch.nn as nn

from einops import rearrange

def att(v, i):
    att = nn.MultiheadAttention(
        embed_dim=32, 
        num_heads=4, 
        dropout=0.0, 
        bias=True, 
        add_bias_kv=False, 
        add_zero_attn=False, 
        kdim=512, 
        vdim=512, 
        batch_first=True, 
        device=None, 
        dtype=None
    )

    attn_output, attn_output_weights = att(v, i, i)

    print(attn_output.size())

if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/tools/debug_2.py
    voxel = torch.randn(1, 32, 128, 32)
    img = torch.randn(1, 512, 128, 32)

    voxel = rearrange(voxel, 'b c h w -> b (h w) c')
    img = rearrange(img, 'b c h w -> b (h w) c')

    att(voxel, img)
    print(voxel.size())
    print(img.size())