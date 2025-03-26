import torch

from mmcv.cnn import ConvModule
from projects.model.tpv.swin import Swin
from configs.config import CONF
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedLSSFPN(nn.Module):

    def __init__(
            self,
            in_channels=[192, 384, 768],
            out_channels=32,
            start_level=0,
            num_outs=3,
            end_level=-1,
            no_norm_on_lateral=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN2d', requires_grad=True, track_running_stats=False),
            act_cfg=dict(type='LeakyReLU', inplace=True),
            upsample_cfg=dict(mode='bilinear', align_corners=False),
            order=("conv", "norm", "act"),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
                order=order,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
                order=order,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    # @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)


class GlobalWeightedAvgPool3D(nn.Module):

    def __init__(self, dim, split, grid_size):
        super(GlobalWeightedAvgPool3D, self).__init__()
        self.dim = dim
        self.split = split
        self.grid_size = grid_size

    def forward(self, x, weights):
        # x : b, c, h, w, z
        # weights : b, 3, h, w, z
        if self.dim == 'xy':
            weight = F.softmax(weights[:, 0:1, :, :, :], dim=-1)
            feat = (x * weight).sum(dim=-1)
        elif self.dim == 'yz':
            weight = F.softmax(weights[:, 1:2, :, :, :], dim=-3)
            feat = (x * weight).sum(dim=-3)
        elif self.dim == 'zx':
            weight = F.softmax(weights[:, 2:3, :, :, :], dim=-2)
            feat = (x * weight).sum(dim=-2)

        return feat


class TPVWeightedAvgPooler(nn.Module):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
    ):
        super().__init__()
        self.weights_conv = nn.Sequential(nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False))
        self.pool_xy = GlobalWeightedAvgPool3D(dim='xy', split=split, grid_size=grid_size)
        self.pool_yz = GlobalWeightedAvgPool3D(dim='yz', split=split, grid_size=grid_size)
        self.pool_zx = GlobalWeightedAvgPool3D(dim='zx', split=split, grid_size=grid_size)
        in_channels = [embed_dims for _ in split]
        out_channels = [embed_dims for _ in split]

        self.mlp_xy = nn.Sequential(nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1))

        self.mlp_yz = nn.Sequential(nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, stride=1))

        self.mlp_zx = nn.Sequential(nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1, stride=1))

    def forward(self, x):
        weights = self.weights_conv(x)
        tpv_xy = self.mlp_xy(self.pool_xy(x, weights))
        tpv_yz = self.mlp_yz(self.pool_yz(x, weights))
        tpv_zx = self.mlp_zx(self.pool_zx(x, weights))

        tpv_list = [tpv_xy, tpv_yz, tpv_zx]

        return tpv_list


class TPVGenerator(nn.Module):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
        pooler='avg',
        global_encoder_backbone=None,
        global_encoder_neck=None,
    ):
        super().__init__()

        # pooling
        if pooler == 'avg':
            self.tpv_pooler = TPVWeightedAvgPooler(embed_dims=embed_dims, split=split, grid_size=grid_size)

        self.global_encoder_backbone = Swin(
            in_channels=embed_dims,
            strides=[1, 2, 2, 2],
            out_indices=[1, 2, 3],
            convert_weights=True,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=CONF.PATH.CKPT_SWIN
            ),
        )

        self.global_encoder_neck = GeneralizedLSSFPN(
            out_channels=embed_dims
        )

        self.grid_size = grid_size

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """

        x_tpv = self.tpv_pooler(x)

        x_tpv = self.global_encoder_backbone(x_tpv)

        tpv_list = []
        neck_out = []
        for view in x_tpv:
            view = self.global_encoder_neck(view)
            neck_out.append(view)
            if not isinstance(view, torch.Tensor):
                view = view[0]
            tpv_list.append(view)

        feats_all = dict()
        feats_all['tpv_backbone'] = x_tpv
        feats_all['tpv_neck'] = neck_out

        # xy
        xy = F.interpolate(tpv_list[0], size=(self.grid_size[0], self.grid_size[1]), mode='bilinear', align_corners=False)
        # yz
        yz = F.interpolate(tpv_list[1], size=(self.grid_size[1], self.grid_size[2]), mode='bilinear', align_corners=False)
        # zx
        xz = F.interpolate(tpv_list[2], size=(self.grid_size[0], self.grid_size[2]), mode='bilinear', align_corners=False)

        return [xy, xz, yz], feats_all


class TPVAggregator(nn.Module):

    def __init__(
        self,
        embed_dims=128,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.combine_coeff = nn.Conv3d(embed_dims, 4, kernel_size=1, bias=False)

    def forward(self, tpv_list, x3d):
        b, c, h, w, z = x3d.size()
        weights = torch.ones([b, 4, h, w, z], device=x3d.device)
        x3d_ = self.weighted_sum([*tpv_list, x3d], weights)
        weights = self.combine_coeff(x3d_)
        out_feats = self.weighted_sum([*tpv_list, x3d], F.softmax(weights, dim=1))

        return out_feats, weights

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats


