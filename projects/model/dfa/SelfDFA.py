import torch
import torch.nn as nn 

from .DFA2D import DeformableAttention2D

from mmcv.cnn.bricks.transformer import build_feedforward_network

from einops import rearrange

class SelfDFA(nn.Module):
    def __init__(
        self,
        geo_feat_channels,
        shape,
        dim_head,
        heads,
        ffn_cfg,
    ):
        super(SelfDFA, self).__init__()
        self.shape = shape

        self.dfa_xy = DeformableAttention2D(
            dim=geo_feat_channels,
            dim_head=dim_head,
            heads=heads,
        )

        self.dfa_xz = DeformableAttention2D(
            dim=geo_feat_channels,
            dim_head=dim_head,
            heads=heads,
        )

        self.dfa_yz = DeformableAttention2D(
            dim=geo_feat_channels,
            dim_head=dim_head,
            heads=heads,
        )

        # Self Aggragation
        self.norm_input_1 =nn.InstanceNorm2d(geo_feat_channels)
        self.norm_input_2 =nn.InstanceNorm2d(geo_feat_channels)
        self.norm_input_3 =nn.InstanceNorm2d(geo_feat_channels)

        self.norm_output_1 =nn.InstanceNorm2d(geo_feat_channels)
        self.norm_output_2 =nn.InstanceNorm2d(geo_feat_channels)
        self.norm_output_3 =nn.InstanceNorm2d(geo_feat_channels)

        # FFN Layer
        self.ffn_xy = build_feedforward_network(ffn_cfg)
        self.ffn_xz = build_feedforward_network(ffn_cfg)
        self.ffn_yz = build_feedforward_network(ffn_cfg)

    def forward(self, tpv_feat):
        # use_residual
        identity_xy = tpv_feat[0].clone()
        identity_xz = tpv_feat[1].clone()
        identity_yz = tpv_feat[2].clone()

        # Norm
        self_xy = self.norm_input_1(tpv_feat[0])
        self_xz = self.norm_input_2(tpv_feat[1])
        self_yz = self.norm_input_3(tpv_feat[2])

        # DeformableSelfAttention2D
        self_xy = self.dfa_xy(self_xy)
        self_xz = self.dfa_xz(self_xz)
        self_yz = self.dfa_yz(self_yz)

        self_xy = self.norm_output_1(self_xy + identity_xy)
        self_xz = self.norm_output_2(self_xz + identity_xz)
        self_yz = self.norm_output_3(self_yz + identity_yz)


        # FFN Rearrange
        identity_xy = rearrange(identity_xy, 'b c x y -> b (x y) c')
        identity_xz = rearrange(identity_xz, 'b c x z -> b (x z) c')
        identity_yz = rearrange(identity_yz, 'b c y z -> b (y z) c')

        self_xy = rearrange(self_xy, 'b c x y -> b (x y) c')
        self_xz = rearrange(self_xz, 'b c x z -> b (x z) c')
        self_yz = rearrange(self_yz, 'b c y z -> b (y z) c')

        # FFN FeadForward
        self_xy = self.ffn_xy(identity_xy, self_xy)
        self_xz = self.ffn_xz(identity_xz, self_xz)
        self_yz = self.ffn_yz(identity_yz, self_yz)

        # FFN output
        self_xy = rearrange(self_xy, 'b (x y) c -> b c x y', x=self.shape[0], y=self.shape[1])
        self_xz = rearrange(self_xz, 'b (x z) c -> b c x z', x=self.shape[0], z=self.shape[2])
        self_yz = rearrange(self_yz, 'b (y z) c -> b c y z', y=self.shape[1], z=self.shape[2])

        return self_xy, self_xz, self_yz

