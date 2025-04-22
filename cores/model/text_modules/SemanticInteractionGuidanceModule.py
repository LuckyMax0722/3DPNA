import torch
import torch.nn as nn

from einops import rearrange

class SemanticInteractionGuidanceModule(nn.Module):
    def __init__(
        self, 
        geo_feat_channels,
        text_dim=512
        ):

        super(SemanticInteractionGuidanceModule, self).__init__()

        self.gamma_fc = nn.Linear(text_dim, geo_feat_channels)
        self.beta_fc  = nn.Linear(text_dim, geo_feat_channels)

    def forward(self, voxel_feat, text_feat):
        """
        fusion_feat: [B, T, H, W, C]
        text_emb:    [B, d]
        """
        B, C, H, W, Z = voxel_feat.shape
        
        voxel_feat = rearrange(voxel_feat, 'b c h w z -> b h w z c')  

        gamma = self.gamma_fc(text_feat)  # [B, C]
        beta  = self.beta_fc(text_feat)   # [B, C]
        
        gamma = gamma.view(B, 1, 1, 1, C)
        beta  = beta.view(B, 1, 1, 1, C)
        
        out = (1 + gamma) * voxel_feat + beta
        
        out = rearrange(out, 'b h w z c -> b c h w z')

        return out