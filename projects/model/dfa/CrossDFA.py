import torch
import torch.nn as nn 

from .DFA2D import DeformableCrossAttention2D

from mmcv.cnn.bricks.transformer import build_feedforward_network

from mmdet.models.backbones.resnet import ResNet

from einops import rearrange

class CrossDFA(nn.Module):
    def __init__(
        self,
        geo_feat_channels,
        img_feat_channels,
        shape,
        dim_head,
        heads,
        ffn_cfg,
    ):
        super(CrossDFA, self).__init__()
        self.shape = shape

        self.img_backbone = ResNet(
            depth=50,
            in_channels=23,
            num_stages=1,
            out_indices=(0,),
            strides=(1,),
            dilations=(1,),
            )

        self.feat_resize_1 = nn.Sequential(
            nn.Conv2d(img_feat_channels, img_feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(img_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True),
            nn.Conv2d(img_feat_channels, img_feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(img_feat_channels),
        )

        self.feat_resize_2 = nn.Sequential(
            nn.Conv2d(img_feat_channels, img_feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(img_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Upsample(size=(128, 32), mode='bilinear', align_corners=True),
            nn.Conv2d(img_feat_channels, img_feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(img_feat_channels),
        )

        self.feat_resize_3 = nn.Sequential(
            nn.Conv2d(img_feat_channels, img_feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(img_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Upsample(size=(128, 32), mode='bilinear', align_corners=True),
            nn.Conv2d(img_feat_channels, img_feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(img_feat_channels),
        )

        self.dfca_xy = DeformableCrossAttention2D(
            dim=geo_feat_channels,
            kv_dim=img_feat_channels,
            dim_head=dim_head,
            heads=heads,
        )

        self.dfca_xz = DeformableCrossAttention2D(
            dim=geo_feat_channels,
            kv_dim=img_feat_channels,
            dim_head=dim_head,
            heads=heads,
        )

        self.dfca_yz = DeformableCrossAttention2D(
            dim=geo_feat_channels,
            kv_dim=img_feat_channels,
            dim_head=dim_head,
            heads=heads,
        )

        # Cross Aggragation
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

    def forward(self, tpv_feat, seg_feat, image):
        # use_residual
        identity_xy = tpv_feat[0].clone()
        identity_xz = tpv_feat[1].clone()
        identity_yz = tpv_feat[2].clone()

        # Norm
        cross_xy = self.norm_input_1(tpv_feat[0])
        cross_xz = self.norm_input_2(tpv_feat[1])
        cross_yz = self.norm_input_3(tpv_feat[2])

        # DeformableCrossAttention2D    
        feat = torch.cat((seg_feat, image), dim=1)
        feat = self.img_backbone(feat)[0]  # torch.Size([1, 256, 96, 320])
        feat = rearrange(feat, 'b c h w -> b c w h')  # torch.Size([1, 256, 320, 96])

        feat_xy = self.feat_resize_1(feat)
        feat_xz = self.feat_resize_2(feat)
        feat_yz = self.feat_resize_3(feat)

        cross_xy = self.dfca_xy(cross_xy, feat_xy)
        cross_xz = self.dfca_xz(cross_xz, feat_xz)
        cross_yz = self.dfca_yz(cross_yz, feat_yz)

        cross_xy = self.norm_output_1(cross_xy + identity_xy)
        cross_xz = self.norm_output_2(cross_xz + identity_xz)
        cross_yz = self.norm_output_3(cross_yz + identity_yz)

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

