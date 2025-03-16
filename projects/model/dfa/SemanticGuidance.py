import torch
import torch.nn as nn 

from einops import rearrange

from .SelfDFA import SelfDFA
from .CrossDFA import CrossDFA


class SemanticGuidanceModule(nn.Module):
    def __init__(
        self,
        geo_feat_channels,
        img_feat_channels,
        shape,
        dim_head,
        heads,
        ffn_cfg,
    ):
        super(SemanticGuidanceModule, self).__init__()

        self.self_dfa = SelfDFA(
            geo_feat_channels=geo_feat_channels,
            shape=shape,
            dim_head=dim_head,
            heads=heads,
            ffn_cfg=ffn_cfg,
        )

        self.cross_dfa = CrossDFA(
            geo_feat_channels=geo_feat_channels,
            img_feat_channels=img_feat_channels,
            shape=shape,
            dim_head=dim_head,
            heads=heads,
            ffn_cfg=ffn_cfg,
        )


    def forward(self, tpv_feat, seg_feat, image):
        self_xy, self_xz, self_yz = self.self_dfa(tpv_feat)

        cross_xy, cross_xz, cross_yz = self.cross_dfa(tpv_feat, seg_feat, image)

        print(cross_xy.size())