import torch
import torch.nn as nn 

from einops import rearrange

from .SelfDFA import SelfDFA
from .CrossDFA import CrossDFA
#from .CrossDFA_V2 import CrossDFAV2

from mmcv.cnn.bricks.transformer import build_feedforward_network

class SemanticGuidanceModule(nn.Module):
    def __init__(
        self,
        geo_feat_channels,
        img_feat_channels,
        kv_dim,
        shape,
        dim_head,
        heads,
        ffn_cfg,
        use_CDFA_v1=True,
    ):
        super(SemanticGuidanceModule, self).__init__()
        self.shape = shape

        self.self_dfa = SelfDFA(
            geo_feat_channels=geo_feat_channels,
            shape=shape,
            dim_head=dim_head,
            heads=heads,
            ffn_cfg=ffn_cfg,
        )

        if use_CDFA_v1:
            self.cross_dfa = CrossDFA(
                geo_feat_channels=geo_feat_channels,
                img_feat_channels=img_feat_channels,
                kv_dim=kv_dim,
                shape=shape,
                dim_head=dim_head,
                heads=heads,
                ffn_cfg=ffn_cfg,
            )
        # else:
        #     self.cross_dfa = CrossDFAV2(
        #         geo_feat_channels=geo_feat_channels,
        #         img_feat_channels=img_feat_channels,
        #         kv_dim=kv_dim,
        #         shape=shape,
        #         dim_head=dim_head,
        #         heads=heads,
        #         ffn_cfg=ffn_cfg,
        #     )

        # Cross Aggragation
        self.norm_input = nn.InstanceNorm2d(geo_feat_channels)
        self.norm_output = nn.InstanceNorm2d(geo_feat_channels)


        # FFN Layer
        self.ffn_xy = build_feedforward_network(ffn_cfg)
        self.ffn_xz = build_feedforward_network(ffn_cfg)
        self.ffn_yz = build_feedforward_network(ffn_cfg)


    def forward(self, tpv_feat, seg_feat):
        self_xy, self_xz, self_yz = self.self_dfa(tpv_feat)

        cross_xy, cross_xz, cross_yz = self.cross_dfa(tpv_feat, seg_feat)

        # Output Norm
        xy = self.norm_input(self_xy + cross_xy)
        xz = self.norm_input(self_xz + cross_xz)
        yz = self.norm_input(self_yz + cross_yz)

        # FFN Rearrange
        xy = rearrange(xy, 'b c x y -> b (x y) c')
        xz = rearrange(xz, 'b c x z -> b (x z) c')
        yz = rearrange(yz, 'b c y z -> b (y z) c')

        # FFN FeadForward
        xy = self.ffn_xy(xy)
        xz = self.ffn_xz(xz)
        yz = self.ffn_yz(yz)

        # FFN output
        xy = rearrange(xy, 'b (x y) c -> b c x y', x=self.shape[0], y=self.shape[1])
        xz = rearrange(xz, 'b (x z) c -> b c x z', x=self.shape[0], z=self.shape[2])
        yz = rearrange(yz, 'b (y z) c -> b c y z', y=self.shape[1], z=self.shape[2])

        # Model Output
        xy = self.norm_output(xy).unsqueeze(-1)
        xz = self.norm_output(xz).unsqueeze(2)
        yz = self.norm_output(yz).unsqueeze(3)

        return [xy, xz, yz]
