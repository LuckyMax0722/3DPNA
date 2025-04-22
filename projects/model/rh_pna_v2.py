import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from projects.loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss

from projects.model.pna_modules.NA_Decoder import NeighborhoodAggregationDecoder as NAD
from projects.model.pna_modules.NA_Encoder import NeighborhoodAggregationEncoder as NAE

class ConvBlock(nn.Module):
    def __init__(
        self, 
        input_channels, 
        output_channels, 
        padding_mode='replicate', 
        stride=(1, 1, 1), 
        kernel_size = (5, 5, 3), 
        padding = (2, 2, 1)
    ):

        super().__init__()
        
        self.convblock = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(input_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(output_channels)
        )
    
    def forward(self, x):
        
        x = self.convblock(x)
        
        return x
    
class ResConvBlock(nn.Module):
    def __init__(
        self, 
        geo_feat_channels, 
        padding_mode='replicate', 
        stride=(1, 1, 1), 
        kernel_size = (5, 5, 3), 
        padding = (2, 2, 1)
    ):

        super().__init__()
        
        self.convblock = ConvBlock(input_channels=geo_feat_channels * 2, output_channels=geo_feat_channels)

            
    def forward(self, skip, x):

        x = torch.cat([x, skip], dim=1)
        x = self.convblock(x)

        return x

class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, z_down=True):
        super().__init__()
        self.z_down = z_down

        self.convblock = ConvBlock(input_channels=geo_feat_channels, output_channels=geo_feat_channels)

        if z_down :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode='replicate'),
                nn.InstanceNorm3d(geo_feat_channels)
            )
    
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        residual_feat = x
        x = self.convblock(x)  # [b, geo_feat_channels, X, Y, Z]
        skip = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
    
        if self.z_down:
            x = self.downsample(skip)  # [b, geo_feat_channels, X//2, Y//2, Z//2]
            return x, skip
        else:
            return skip  # [b, geo_feat_channels, X, Y, Z]

class Decoder(nn.Module):
    def __init__(self, geo_feat_channels):
        super().__init__()
        
        self.convblock = ResConvBlock(geo_feat_channels=geo_feat_channels)
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, x, skip):
        
        x = self.up_scale(x)
        x = self.convblock(skip, x)
        
        return x

class Header(nn.Module):
    def __init__(
        self,
        geo_feat_channels,
        class_num
    ):
        super(Header, self).__init__()

        self.conv_head = nn.Conv3d(geo_feat_channels, class_num, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [1, 64, 256, 256, 32]

        ssc_logit = self.conv_head(x)
   
        return ssc_logit

class UNet(nn.Module):
    def __init__(self, 
        geo_feat_channels,
       
        ffn_cfg,
        num_heads,
        kernel_size,
        dilation,
        rel_pos_bias,
        qkv_bias,
        attn_drop,
        proj_drop,
        use_fna
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
            geo_feat_channels
        )

        self.pna_block_encoder_128 = NAE(
            embed_dims=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[2],
            dilation=dilation[2],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )

        self.pna_block_encoder_64 = NAE(
            embed_dims=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[1],
            dilation=dilation[1],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )

        self.pna_block_encoder_32 = NAE(
            embed_dims=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[0],
            dilation=dilation[0],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )

        self.bottleneck = Encoder(
            geo_feat_channels,
            z_down=False
        )

        self.pna_block_decoder_32 = NAD(
            embed_dims=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[0],
            dilation=dilation[0],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )

        self.pna_block_decoder_64 = NAD(
            embed_dims=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[1],
            dilation=dilation[1],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )

        self.pna_block_decoder_128 = NAD(
            embed_dims=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size[2],
            dilation=dilation[2],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
        )

        self.decoder_block_1 = Decoder(
            geo_feat_channels, 
        )

        
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]   
        
        x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
        
        x, skip_256 = self.encoder_block_1(x) # skip1: ([1, 64, 256, 256, 32]) / x: ([1, 64, 128, 128, 16])
        
        x, skip_128 = self.pna_block_encoder_128(x) # skip2: ([1, 64, 128, 128, 16]) / x: ([1, 64, 64, 64, 8])
        
        x, skip_64 = self.pna_block_encoder_64(x) # skip3: ([1, 64, 64, 64, 8]) / x: ([1, 64, 32, 32, 4])
        
        x, skip_32 = self.pna_block_encoder_32(x) # skip4: ([1, 64, 32, 32, 4]) / x: ([1, 64, 16, 16, 2])
        
        x_16 = self.bottleneck(x) # x: ([1, 64, 16, 16, 2])
        
        x_32 = self.pna_block_decoder_32(x_16, skip_32)  # x: ([1, 64, 32, 32, 4])
        
        x_64 = self.pna_block_decoder_64(x_32, skip_64)  # x: ([1, 64, 64, 64, 8]) 
        
        x_128 = self.pna_block_decoder_128(x_64, skip_128)  # x: ([1, 64, 128, 128, 16])
        
        x_256 = self.decoder_block_1(x_128, skip_256)  # x: ([1, 64, 256, 256, 32])
        
        return x_32, x_64, x_128, x_256

class PredHead(nn.Module):
    def __init__(self, 
        geo_feat_channels,
        num_class
    ):

        super().__init__()

        self.pred_head_256 = Header(geo_feat_channels, num_class)
        self.pred_head_128 = Header(geo_feat_channels, num_class)
        self.pred_head_64 = Header(geo_feat_channels, num_class)
        self.pred_head_32 = Header(geo_feat_channels, num_class)

    def forward(self, x_32, x_64, x_128, x_256):  # [b, geo_feat_channels, X, Y, Z]   

        x_256 = self.pred_head_256(x_256)
        x_128 = self.pred_head_128(x_128)     
        x_64 = self.pred_head_64(x_64)   
        x_32 = self.pred_head_32(x_32)

        return x_32, x_64, x_128, x_256

class RefHead_PNA_V2(nn.Module):
    def __init__(
        self,
        num_class=None,
        geo_feat_channels=None,
        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,

        ffn_cfg=None,
        num_heads=None,
        kernel_size=None,
        dilation=None,
        rel_pos_bias=None,
        qkv_bias=None,
        attn_drop=None,
        proj_drop=None,
        use_fna=None,
    ):
        super(RefHead_PNA_V2, self).__init__()
        
        self.empty_idx = empty_idx
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        
        self.unet = UNet(
            geo_feat_channels=geo_feat_channels,
            ffn_cfg=ffn_cfg,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_fna=use_fna,
            )
        
        self.pred_head = PredHead(
            geo_feat_channels=geo_feat_channels,
            num_class=num_class
        )
        
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

        x_32, x_64, x_128, x_256 = self.unet(x)
        
        x_32, x_64, x_128, x_256 = self.pred_head(x_32, x_64, x_128, x_256)
        
        return x_32, x_64, x_128, x_256
    
    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [32, 64, 128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict

if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_pna_v2.py
    
    from configs.config import CONF
    from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset


    ds = SemanticKITTIDataset(
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model='CGFormer',
        text_model=None,
        split='train'
        )

    voxel = ds[0]['input_occ'].unsqueeze(0).cuda()
   
    kernel_size = [[3,3,3], [5,5,5], [7,7,7]]

    dilation = [[1,1,1], [1,1,1], [1,1,1]]

    rh = RefHead_PNA_V2(
        num_class=20,
        geo_feat_channels=32,
        balance_cls_weight=True,
        class_frequencies=[
            5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
            8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
            4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
            1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
        ],

        
        ffn_cfg=dict(
            type='FFN',
            embed_dims=32,
            feedforward_channels=512,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.1,
            add_identity=True
        ),

        num_heads=4,
        kernel_size=kernel_size,
        dilation=dilation,
        rel_pos_bias=True,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        use_fna=False,


    ).cuda()

    x_32, x_64, x_128, x_256 = rh(voxel)

    print(x_256.size())