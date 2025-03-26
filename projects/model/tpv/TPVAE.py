import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Encoder, Header, Decoder

class TPVAE(nn.Module):
    def __init__(
        self, 
        num_class,
        geo_feat_channels,
        z_down,
        padding_mode
        ) -> None:

        super().__init__()

        self.geo_feat_dim = geo_feat_channels

        # Encoder
        self.embedding = nn.Embedding(num_class, geo_feat_channels)

        self.conv_in = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(5, 5, 3), stride=(1, 1, 1), padding=(2, 2, 1), bias=True, padding_mode=padding_mode)

        self.geo_encoder_128 = Encoder(geo_feat_channels, z_down, padding_mode)
        self.geo_encoder_64 = Encoder(geo_feat_channels, z_down, padding_mode)
        self.geo_encoder_32 = Encoder(geo_feat_channels, z_down, padding_mode)

        self.norm = nn.InstanceNorm2d(geo_feat_channels)

        # Decoder
        self.combine_coeff_32 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, 4, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )


        self.decoder_64 = Decoder(
            geo_feat_channels=geo_feat_channels
        )

        self.decoder_128 = Decoder(
            geo_feat_channels=geo_feat_channels
        )

        self.decoder_256 = Decoder(
            geo_feat_channels=geo_feat_channels
        )

        self.pred_head_256 = Header(geo_feat_channels, num_class)
        self.pred_head_128 = Header(geo_feat_channels, num_class)
        self.pred_head_64 = Header(geo_feat_channels, num_class)
        self.pred_head_32 = Header(geo_feat_channels, num_class)

    def to_tpv(self, x):
        xy_feat = x.mean(dim=4)
        xz_feat = x.mean(dim=3)
        yz_feat = x.mean(dim=2)
        
        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]

    def encode(self, vol):
        x = vol.detach().clone()
        x[x == 255] = 0
            
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)

        x_256 = self.conv_in(x)

        x_128 = self.geo_encoder_128(x_256)
        x_64 = self.geo_encoder_64(x_128)
        x_32 = self.geo_encoder_32(x_64)

        tpv_x_32 = self.to_tpv(x_32)
        tpv_x_64 = self.to_tpv(x_64)
        tpv_x_128 = self.to_tpv(x_128)
        tpv_x_256 = self.to_tpv(x_256)

        return [tpv_x_256, tpv_x_128, tpv_x_64, tpv_x_32], [x_256, x_128, x_64, x_32]
    
    def encoder(self, vol):
        '''
        Output:
            feat_map: [
                ...
                [
                    torch.Size([1, 32, 64, 64])
                    torch.Size([1, 32, 64, 8])
                    torch.Size([1, 32, 64, 8])
                ],
                [
                    torch.Size([1, 32, 32, 32])
                    torch.Size([1, 32, 32, 4])
                    torch.Size([1, 32, 32, 4])
                ]
            ]
            vol_feat_map: [
                torch.Size([1, 32, 256, 256, 32])
                torch.Size([1, 32, 128, 128, 16])
                torch.Size([1, 32, 64, 64, 8])
                torch.Size([1, 32, 32, 32, 4])
            ]
        '''
        feat_map, vol_feat_map = self.encode(vol)

        return feat_map, vol_feat_map

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats

    def decode(self, feat_maps, vol_feat_maps):

        # CGFormer

        weights_32 = self.combine_coeff_32(vol_feat_maps[3])

        x_32 = vol_feat_maps[3] * weights_32[:, 0:1, ...] + feat_maps[3][0] * weights_32[:, 1:2, ...] + \
            feat_maps[3][1] * weights_32[:, 2:3, ...] + feat_maps[3][2] * weights_32[:, 3:4, ...]  # torch.Size([1, 32, 32, 32, 4])


        # weights_64 = self.combine_coeff_64(vol_feat_maps[2])

        # x_64 = vol_feat_maps[2] * weights_64[:, 0:1, ...] + feat_maps[2][0] * weights_64[:, 1:2, ...] + \
        #     feat_maps[2][1] * weights_64[:, 2:3, ...] + feat_maps[2][2] * weights_64[:, 3:4, ...]

        # weights_128 = self.combine_coeff_128(vol_feat_maps[1])

        # x_128 = vol_feat_maps[1] * weights_128[:, 0:1, ...] + feat_maps[1][0] * weights_128[:, 1:2, ...] + \
        #     feat_maps[1][1] * weights_128[:, 2:3, ...] + feat_maps[1][2] * weights_128[:, 3:4, ...]

        # x_64 = self.decoder_64(x_64, x_32)
        # x_128 = self.decoder_128(x_128, x_64)
        
        # # L2COcc
        # b, c, h, w, z = vol_feat_maps[3].size()
        # weights = torch.ones([b, 4, h, w, z], device=vol_feat_maps[3].device)
        # x3d_ = self.weighted_sum([*feat_maps, vol_feat_maps[3]], weights)
        # weights = self.combine_coeff(x3d_)
        # x_32 = self.weighted_sum([*feat_maps, vol_feat_maps[3]], F.softmax(weights, dim=1))

        x_64 = self.decoder_64(vol_feat_maps[2], x_32)
        x_128 = self.decoder_128(vol_feat_maps[1], x_64)
        x_256 = self.decoder_256(vol_feat_maps[0], x_128)

        return x_32, x_64, x_128, x_256

    def decoder(self, feat_maps, vol_feat_maps):

        x_32, x_64, x_128, x_256 = self.decode(feat_maps, vol_feat_maps)

        x_256 = self.pred_head_256(x_256)
        x_128 = self.pred_head_128(x_128)
        x_64 = self.pred_head_64(x_64)
        x_32 = self.pred_head_32(x_32)

        return x_32, x_64, x_128, x_256
