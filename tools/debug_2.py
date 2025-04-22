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
    import numpy as np
    text_f = np.load('/u/home/caoh/datasets/SemanticKITTI/dataset/text/feat/CLIP/00/000000.npy')
    print(text_f.dtype)