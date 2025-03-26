import torch
import torch.nn as nn 

from .DFA2D import DeformableCrossAttention2D

from mmcv.cnn.bricks.transformer import build_feedforward_network

from einops import rearrange

class CrossDFAV2(nn.Module):
    def __init__(
        self,
        geo_feat_channels,
        img_feat_channels,
        kv_dim,
        shape,
        dim_head,
        heads,
        ffn_cfg,
    ):
        super(CrossDFAV2, self).__init__()
        self.shape = shape


        self.dfca_xy = DeformableCrossAttention2D(
            dim=geo_feat_channels,
            kv_dim=kv_dim,
            dim_head=dim_head,
            heads=heads,
        )

        self.dfca_xz = DeformableCrossAttention2D(
            dim=geo_feat_channels,
            kv_dim=kv_dim,
            dim_head=dim_head,
            heads=heads,
        )

        self.dfca_yz = DeformableCrossAttention2D(
            dim=geo_feat_channels,
            kv_dim=kv_dim,
            dim_head=dim_head,
            heads=heads,
        )

        # Cross Aggragation
        self.norm_input =nn.InstanceNorm2d(geo_feat_channels)
        self.norm_output =nn.InstanceNorm2d(geo_feat_channels)

        # FFN Layer
        self.ffn_xy = build_feedforward_network(ffn_cfg)
        self.ffn_xz = build_feedforward_network(ffn_cfg)
        self.ffn_yz = build_feedforward_network(ffn_cfg)

    def forward(self, tpv_feat, seg_feat):
        # use_residual
        identity_xy = tpv_feat[0]
        identity_xz = tpv_feat[1]
        identity_yz = tpv_feat[2]

        # Norm
        cross_xy = self.norm_input(tpv_feat[0])
        cross_xz = self.norm_input(tpv_feat[1])
        cross_yz = self.norm_input(tpv_feat[2])

        # DeformableCrossAttention2D    
        
        feat = rearrange(seg_feat, 'b c h w -> b c w h')  # torch.Size([1, 256, 320, 96])

        feat_xy = self.feat_resize_1(feat)
        feat_xz = self.feat_resize_2(feat)
        feat_yz = self.feat_resize_3(feat)

        cross_xy = self.dfca_xy(cross_xy, feat_xy)
        cross_xz = self.dfca_xz(cross_xz, feat_xz)
        cross_yz = self.dfca_yz(cross_yz, feat_yz)

        cross_xy = self.norm_output(cross_xy + identity_xy)
        cross_xz = self.norm_output(cross_xz + identity_xz)
        cross_yz = self.norm_output(cross_yz + identity_yz)

        # FFN Rearrange
        identity_xy = rearrange(identity_xy, 'b c x y -> b (x y) c')
        identity_xz = rearrange(identity_xz, 'b c x z -> b (x z) c')
        identity_yz = rearrange(identity_yz, 'b c y z -> b (y z) c')

        cross_xy = rearrange(cross_xy, 'b c x y -> b (x y) c')
        cross_xz = rearrange(cross_xz, 'b c x z -> b (x z) c')
        cross_yz = rearrange(cross_yz, 'b c y z -> b (y z) c')

        # FFN FeadForward
        cross_xy = self.ffn_xy(identity_xy, cross_xy)
        cross_xz = self.ffn_xz(identity_xz, cross_xz)
        cross_yz = self.ffn_yz(identity_yz, cross_yz)

        # FFN output
        cross_xy = rearrange(cross_xy, 'b (x y) c -> b c x y', x=self.shape[0], y=self.shape[1])
        cross_xz = rearrange(cross_xz, 'b (x z) c -> b c x z', x=self.shape[0], z=self.shape[2])
        cross_yz = rearrange(cross_yz, 'b (y z) c -> b c y z', y=self.shape[1], z=self.shape[2])

        return cross_xy, cross_xz, cross_yz

