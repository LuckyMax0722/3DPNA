import torch
import torch.nn as nn
import numpy as np

from projects.loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from .pna import ProgressiveNeighborhoodAggregation as PNA

class ConvBlock(nn.Module):
    def __init__(self, geo_feat_channels, padding_mode='replicate', stride=(1, 1, 1), kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        
        self.convblock = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
    
    def forward(self, x):
        
        x = self.convblock(x)
        
        return x
    
class ResConvBlock(nn.Module):
    def __init__(self, 
                 geo_feat_channels, 
                 version,
                 padding_mode='replicate', 
                 stride=(1, 1, 1), 
                 kernel_size = (5, 5, 3), 
                 padding = (2, 2, 1)):
        super().__init__()
        
        self.version = version
        
        if self.version == 'plus':
            self.convblock = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels),
                nn.LeakyReLU(1e-1, True),
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        
        if self.version == 'concat':
            self.convblock = nn.Sequential(
                nn.Conv3d(geo_feat_channels * 2, geo_feat_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels * 2),
                nn.LeakyReLU(1e-1, True),
                nn.Conv3d(geo_feat_channels * 2, geo_feat_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
            
    def forward(self, skip, x):
        if self.version == 'plus':
            x = skip + x
    
        elif self.version == 'concat':
            x = torch.cat([x, skip], dim=1)
        
        x = self.convblock(x)
        return x

class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, z_down):
        super().__init__()
        self.z_down = z_down
        
        self.convblock = ConvBlock(geo_feat_channels=geo_feat_channels)
        
        if z_down :
            self.downsample = nn.MaxPool3d((2, 2, 2))
    
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        residual_feat = x
        x = self.convblock(x)  # [b, geo_feat_channels, X, Y, Z]
        skip = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
        
        if self.z_down:
            x = self.downsample(skip)  # [b, geo_feat_channels, X//2, Y//2, Z//2]
            return skip, x
        else:
            return skip  # [b, geo_feat_channels, X, Y, Z]

class Decoder(nn.Module):
    def __init__(self, geo_feat_channels):
        super().__init__()
        
        self.convblock = ResConvBlock(geo_feat_channels=geo_feat_channels, version='plus')
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
        version
    ):
        super(Header, self).__init__()
        self.geo_feat_channels = geo_feat_channels
        self.class_num = class_num
        self.version = version
        
        if self.version == 'mlp':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.geo_feat_channels),
                nn.Linear(self.geo_feat_channels, self.class_num),
            )
        elif self.version == 'conv':
            self.conv_head = nn.Sequential(
                nn.Conv3d(self.geo_feat_channels, self.class_num, kernel_size=1, stride=1, padding=0, bias=False),
            )

    def forward(self, x):
        # [1, 64, 256, 256, 32]
        res = {} 

        if self.version == 'mlp':
            _, feat_dim, w, l, h  = x.shape

            x = x.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

            ssc_logit_full = self.mlp_head(x)

            ssc_logit = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)
            
        elif self.version == 'conv':
            
            ssc_logit = self.conv_head(x)
        
        res["ssc_logit"] = ssc_logit
        
        return res


class UNet(nn.Module):
    def __init__(self, 
                use_fna,
                geo_feat_channels,
                num_heads,
                ffn_cfg,
                kernel_size
                ):
        super().__init__()
        
        self.conv0 = nn.Conv3d(geo_feat_channels, 
                               geo_feat_channels, 
                               kernel_size=(5, 5, 3), 
                               stride=(1, 1, 1), 
                               padding=(2, 2, 1), 
                               bias=True, 
                               padding_mode='replicate')
        
        self.encoder_block_1 = Encoder(geo_feat_channels, 
                                       z_down=True
                                       )
        
        self.encoder_block_2 = Encoder(geo_feat_channels, 
                                       z_down=True
                                       )
        
        self.encoder_block_3 = Encoder(geo_feat_channels, 
                                       z_down=True
                                       )

        self.encoder_block_4 = Encoder(geo_feat_channels, 
                                       z_down=True
                                       )
        
        self.bottleneck = Encoder(geo_feat_channels, 
                                  z_down=False
                                  )

        self.pna_block_4 = PNA(
            use_fna=use_fna,
            embed_dims=geo_feat_channels, 
            num_heads=num_heads,
            ffn_cfg = ffn_cfg,
            kernel_size=kernel_size[0]
        )

        self.pna_block_3 = PNA(
            use_fna=use_fna,
            embed_dims=geo_feat_channels, 
            num_heads=num_heads,
            ffn_cfg = ffn_cfg,
            kernel_size=kernel_size[1]
        )

        self.pna_block_2 = PNA(
            use_fna=use_fna,
            embed_dims=geo_feat_channels, 
            num_heads=num_heads,
            ffn_cfg = ffn_cfg,
            kernel_size=kernel_size[2]
        )

        self.decoder_block_1 = Decoder(geo_feat_channels)
        
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]   
        
        x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
        
        skip1, x = self.encoder_block_1(x) # skip1: ([1, 64, 256, 256, 32]) / x: ([1, 64, 128, 128, 16])
        
        skip2, x = self.encoder_block_2(x) # skip2: ([1, 64, 128, 128, 16]) / x: ([1, 64, 64, 64, 8])
        
        skip3, x = self.encoder_block_3(x) # skip3: ([1, 64, 64, 64, 8]) / x: ([1, 64, 32, 32, 4])
        
        skip4, x = self.encoder_block_4(x) # skip4: ([1, 64, 32, 32, 4]) / x: ([1, 64, 16, 16, 2])
        
        x = self.bottleneck(x)
        
        x4 = self.pna_block_4(x, skip4)
        x3 = self.pna_block_3(x4, skip3)
        x2 = self.pna_block_2(x3, skip2)
        
        x1 = self.decoder_block_1(skip1, x2)
        
        return x4, x3, x2, x1

class RefHeadV2(nn.Module):
    def __init__(
        self,
        use_fna,
        num_class,
        geo_feat_channels,
        ffn_cfg,
        kernel_size,
        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
        num_heads=8
    ):
        super(RefHeadV2, self).__init__()
        
        self.empty_idx = empty_idx
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        
        self.unet = UNet(
            use_fna=use_fna,
            geo_feat_channels=geo_feat_channels,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,
            kernel_size=kernel_size
            )
        
        self.pred_head_8 = Header(geo_feat_channels, num_class, version='conv')
        
        self.pred_head_4 = Header(geo_feat_channels, num_class, version='conv')
        
        self.pred_head_2 = Header(geo_feat_channels, num_class, version='conv')
        
        self.pred_head_1 = Header(geo_feat_channels, num_class, version='conv')
        
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
        #x = vol.detach().clone()
        x[x == 255] = 0

        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)

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
    