import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from projects.loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.model.ASPP3D import ASPP3D

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, conv_version, padding_mode='replicate', stride=(1, 1, 1), kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        
        if conv_version == 'v1':
            self.convblock = nn.Sequential(
                nn.Conv3d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(input_channels),
                nn.LeakyReLU(1e-1, True),
                nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(output_channels)
            )
        elif conv_version == 'v2':
            self.convblock = nn.Sequential(
                nn.Conv3d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(input_channels),
                nn.LeakyReLU(1e-1, True),
                nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(output_channels),
                nn.LeakyReLU(1e-1, True)
            )
    
    def forward(self, x):
        
        x = self.convblock(x)
        
        return x
    
class ResConvBlock(nn.Module):
    def __init__(self, 
                 geo_feat_channels, 
                 skip_version,
                 conv_version,
                 decoder_version,
                 shape,
                 padding_mode='replicate', 
                 stride=(1, 1, 1), 
                 kernel_size = (5, 5, 3), 
                 padding = (2, 2, 1)):
        super().__init__()
        
        self.skip_version = skip_version
        
        if self.skip_version == 'plus':

            if decoder_version == 'conv':
                self.convblock = ConvBlock(input_channels=geo_feat_channels, output_channels=geo_feat_channels, conv_version=conv_version)
            elif decoder_version == 'aspp':
                self.convblock = ASPP3D(in_channels=geo_feat_channels, out_channels=geo_feat_channels, shape=shape)

        if self.skip_version == 'concat':

            self.convblock = ConvBlock(input_channels=geo_feat_channels * 2, output_channels=geo_feat_channels, conv_version=conv_version)

            
    def forward(self, skip, x):
        if self.skip_version == 'plus':
            x = skip + x
    
        elif self.skip_version == 'concat':
            x = torch.cat([x, skip], dim=1)
        
        x = self.convblock(x)
        return x

class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, conv_version, encoder_version, z_down, use_skip, shape=None):
        super().__init__()
        self.z_down = z_down
        self.use_skip = use_skip

        if encoder_version == 'conv':
            self.convblock = ConvBlock(input_channels=geo_feat_channels, output_channels=geo_feat_channels, conv_version='v1')
        elif encoder_version == 'aspp':
            self.convblock = ASPP3D(in_channels=geo_feat_channels, out_channels=geo_feat_channels, shape=shape)

        if z_down :
            self.downsample = nn.MaxPool3d((2, 2, 2))
    
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]

        if self.use_skip:
            residual_feat = x
            x = self.convblock(x)  # [b, geo_feat_channels, X, Y, Z]
            skip = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
        else:
            skip = self.convblock(x)
        
        if self.z_down:
            x = self.downsample(skip)  # [b, geo_feat_channels, X//2, Y//2, Z//2]
            return skip, x
        else:
            return skip  # [b, geo_feat_channels, X, Y, Z]

class Decoder(nn.Module):
    def __init__(self, geo_feat_channels, skip_version, conv_version, decoder_version, shape=None):
        super().__init__()
        
        self.convblock = ResConvBlock(geo_feat_channels=geo_feat_channels, skip_version=skip_version, conv_version=conv_version, decoder_version=decoder_version, shape=shape)
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, skip, x):
        
        x = self.up_scale(x)
        x = self.convblock(skip, x)
        
        return x

class Header(nn.Module): # mlp as perdition head
    def __init__(
        self,
        geo_feat_channels,
        class_num,
        head_version
    ):
        super(Header, self).__init__()
        self.geo_feat_channels = geo_feat_channels
        self.class_num = class_num
        self.head_version = head_version
        
        if self.head_version == 'mlp':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.geo_feat_channels),
                nn.Linear(self.geo_feat_channels, self.class_num),
            )
        elif self.head_version == 'conv':
            self.conv_head = nn.Sequential(
                nn.Conv3d(self.geo_feat_channels, self.class_num, kernel_size=1, stride=1, padding=0, bias=False),
            )

    def forward(self, x):
        # [1, 64, 256, 256, 32]
        res = {} 

        if self.head_version == 'mlp':
            _, feat_dim, w, l, h  = x.shape

            x = x.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

            ssc_logit_full = self.mlp_head(x)

            ssc_logit = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)
            
        elif self.head_version == 'conv':
            
            ssc_logit = self.conv_head(x)
        
        res["ssc_logit"] = ssc_logit
        
        return res


class UNet(nn.Module):
    def __init__(self, 
                geo_feat_channels,
                skip_version,
                conv_version,
                encoder_version,
                use_skip,
                decoder_version='conv'
                ):
        super().__init__()
        
        self.conv0 = nn.Conv3d(
            geo_feat_channels, 
            geo_feat_channels, 
            kernel_size=(5, 5, 3), 
            stride=(1, 1, 1), 
            padding=(2, 2, 1), 
            bias=True, 
            padding_mode='replicate'
        )
        
        self.encoder_block_1 = Encoder(
            geo_feat_channels,
            conv_version = conv_version,
            encoder_version=encoder_version,
            shape=[256,256,32],
            use_skip=use_skip,
            z_down=True
        )

        self.encoder_block_2 = Encoder(
            geo_feat_channels,
            conv_version = conv_version,
            encoder_version=encoder_version,
            shape=[128,128,16],
            use_skip=use_skip,
            z_down=True
        )

        self.encoder_block_3 = Encoder(
            geo_feat_channels,
            conv_version = conv_version,
            encoder_version=encoder_version,
            shape=[64,64,8],
            use_skip=use_skip,
            z_down=True
        )

        self.encoder_block_4 = Encoder(
            geo_feat_channels,
            conv_version = conv_version,
            encoder_version=encoder_version,
            shape=[32,32,4],
            use_skip=use_skip,
            z_down=True
        )

        self.bottleneck = Encoder(
            geo_feat_channels,
            conv_version = conv_version,
            encoder_version='conv',
            use_skip=use_skip,
            z_down=False
        )

        self.decoder_block_4 = Decoder(
            geo_feat_channels, 
            skip_version=skip_version, 
            conv_version=conv_version,
            decoder_version=decoder_version,
            shape=[32,32,4]
        )

        self.decoder_block_3 = Decoder(
            geo_feat_channels, 
            skip_version=skip_version, 
            conv_version=conv_version,
            decoder_version=decoder_version,
            shape=[64,64,8]
        )

        self.decoder_block_2 = Decoder(
            geo_feat_channels, 
            skip_version=skip_version, 
            conv_version=conv_version,
            decoder_version=decoder_version,
            shape=[128,128,16]
        )

        self.decoder_block_1 = Decoder(
            geo_feat_channels, 
            skip_version=skip_version, 
            conv_version=conv_version,
            decoder_version=decoder_version,
            shape=[256,256,32]
        )

        
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]   
        
        x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
        
        skip1, x = self.encoder_block_1(x) # skip1: ([1, 64, 256, 256, 32]) / x: ([1, 64, 128, 128, 16])
        
        skip2, x = self.encoder_block_2(x) # skip2: ([1, 64, 128, 128, 16]) / x: ([1, 64, 64, 64, 8])
        
        skip3, x = self.encoder_block_3(x) # skip3: ([1, 64, 64, 64, 8]) / x: ([1, 64, 32, 32, 4])
        
        skip4, x = self.encoder_block_4(x) # skip4: ([1, 64, 32, 32, 4]) / x: ([1, 64, 16, 16, 2])
        
        x = self.bottleneck(x) # x: ([1, 64, 16, 16, 2])
        
        x4 = self.decoder_block_4(skip4, x)  # x: ([1, 64, 32, 32, 4])
        
        x3 = self.decoder_block_3(skip3, x4)  # x: ([1, 64, 64, 64, 8]) 
        
        x2 = self.decoder_block_2(skip2, x3)  # x: ([1, 64, 128, 128, 16])
        
        x1 = self.decoder_block_1(skip1, x2)  # x: ([1, 64, 256, 256, 32])
        
        return x4, x3, x2, x1

class RefHead(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,
        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
        skip_version='plus',
        conv_version='v1',
        encoder_version='conv',
        head_version='conv',
        use_skip=None
    ):
        super(RefHead, self).__init__()
        
        self.empty_idx = empty_idx
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        
        self.unet = UNet(
            geo_feat_channels=geo_feat_channels,
            skip_version=skip_version,
            conv_version=conv_version,
            encoder_version=encoder_version,
            use_skip=use_skip
            )
        
        self.pred_head_8 = Header(geo_feat_channels, num_class, head_version=head_version)
        self.pred_head_4 = Header(geo_feat_channels, num_class, head_version=head_version)
        self.pred_head_2 = Header(geo_feat_channels, num_class, head_version=head_version)
        self.pred_head_1 = Header(geo_feat_channels, num_class, head_version=head_version)
        
        # voxel losses
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
            
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17
            
    def forward(self, x):
        x[x == 255] = 0

        x = self.embedding(x)

        x = rearrange(x, 'b h w z c -> b c h w z ')  

        x8, x4, x2, x1 = self.unet(x)
        
        x8 = self.pred_head_8(x8)
        
        x4 = self.pred_head_4(x4)
        
        x2 = self.pred_head_2(x2)
        
        x1 = self.pred_head_1(x1)
        
        return x1, x2, x4, x8
    
    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [1, 2, 4, 8]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict
    