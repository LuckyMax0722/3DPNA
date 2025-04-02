import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from mmcv.cnn.bricks.transformer import build_feedforward_network

from projects.loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss

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
        
        self.convblock = ResConvBlock(geo_feat_channels=geo_feat_channels)
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, skip, x):
        
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
        ffn_cfg
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

        self.encoder_block_2 = Encoder(
            geo_feat_channels
        )

        self.encoder_block_3 = Encoder(
            geo_feat_channels
        )

        self.encoder_block_4 = Encoder(
            geo_feat_channels
        )

        self.bottleneck = Encoder(
            geo_feat_channels,
            z_down=False
        )

        self.decoder_block_4 = Decoder(
            geo_feat_channels
        )

        self.decoder_block_3 = Decoder(
            geo_feat_channels
        )

        self.decoder_block_2 = Decoder(
            geo_feat_channels
        )

        self.decoder_block_1 = Decoder(
            geo_feat_channels
        )

        # self.crosstextattn_16 = TextAttention(
        #     geo_feat_channels,
        #     ffn_cfg
        # )

        # self.crosstextattn_32 = TextAttention(
        #     geo_feat_channels,
        #     ffn_cfg
        # )

        # self.crosstextattn_64 = TextAttention(
        #     geo_feat_channels,
        #     ffn_cfg
        # )

        # self.crosstextattn_128 = TextAttention(
        #     geo_feat_channels,
        #     ffn_cfg
        # )



    def forward(self, x, text):  # [b, geo_feat_channels, X, Y, Z]   
        
        x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
        
        skip_256, x = self.encoder_block_1(x) # skip1: ([1, 64, 256, 256, 32]) / x: ([1, 64, 128, 128, 16])
        
        skip_128, x = self.encoder_block_2(x) # skip2: ([1, 64, 128, 128, 16]) / x: ([1, 64, 64, 64, 8])
        
        skip_64, x = self.encoder_block_3(x) # skip3: ([1, 64, 64, 64, 8]) / x: ([1, 64, 32, 32, 4])
        
        skip_32, x = self.encoder_block_4(x) # skip4: ([1, 64, 32, 32, 4]) / x: ([1, 64, 16, 16, 2])
        
        x_16 = self.bottleneck(x) # x: ([1, 64, 16, 16, 2])
        
        #x_16 = self.crosstextattn_16(x_16, text)

        x_32 = self.decoder_block_4(skip_32, x_16)  # x: ([1, 64, 32, 32, 4])
        
        #x_32 = self.crosstextattn_32(x_32, text)

        x_64 = self.decoder_block_3(skip_64, x_32)  # x: ([1, 64, 64, 64, 8]) 
        
        #x_64 = self.crosstextattn_64(x_64, text)

        x_128 = self.decoder_block_2(skip_128, x_64)  # x: ([1, 64, 128, 128, 16])
        
        #x_128 = self.crosstextattn_128(x_128, text)

        x_256 = self.decoder_block_1(skip_256, x_128)  # x: ([1, 64, 256, 256, 32])
        
        return x_32, x_64, x_128, x_256


class TextAttention(nn.Module):
    def __init__(self, 
        geo_feat_channels,
        ffn_cfg,
        text_feat_channels = 256,
        num_heads = 4
    ):
        super(TextAttention, self).__init__()

        embed_dim = geo_feat_channels

        # Norm Input
        self.norm_input_text =nn.LayerNorm(text_feat_channels)
        self.norm_input_voxel =nn.InstanceNorm1d(embed_dim)

        # CrossAttention
        self.crossattn = nn.MultiheadAttention(
            embed_dim = embed_dim, 
            num_heads = num_heads, 
            bias=True, 
            kdim=text_feat_channels, 
            vdim=text_feat_channels, 
            batch_first=True, 
        )

        # Norm Output
        self.norm_output_voxel =nn.InstanceNorm1d(embed_dim)

        # FFN Layer
        self.ffn = build_feedforward_network(ffn_cfg)


    def forward(self, x, text):
        '''
        Input:
            x: torch.size: [1, c, x, y, z]
            text: torch.size: [1, c, 256]
        '''
        bs, c, h, w, z = x.shape
        x = rearrange(x, 'b c h w z -> b (h w z) c')  # torch.Size([1, h * w * z, 32])

        # use_residual
        identity_x = x
        identity_text = text

        # Norm
        x = self.norm_input_voxel(x)
        text = self.norm_input_text(text)

        # CrossAttention
        attn_output, attn_output_weights = self.crossattn(query = x, key = text, value = text)

        # Norm
        x = self.norm_output_voxel(attn_output + x)

        # FFN
        x = self.ffn(x, identity_x)

        # FFN output
        x = rearrange(x, 'b (h w z) c -> b c h w z', h=h, w=w, z=z)

        return x


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

class RefHead_Text(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,
        ffn_cfg,

        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,

    ):
        super(RefHead_Text, self).__init__()
        
        self.empty_idx = empty_idx
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        
        self.unet = UNet(
            geo_feat_channels=geo_feat_channels,
            ffn_cfg=ffn_cfg
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
            
    def forward(self, x, text):
        x[x == 255] = 0

        x = self.embedding(x)

        x = rearrange(x, 'b h w z c -> b c h w z')  

        x_32, x_64, x_128, x_256 = self.unet(x, text)
        
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
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_text.py
    
    from configs.config import CONF
    from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset

    ds = SemanticKITTIDataset(
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model='CGFormer',
        text_model='Blip2',
        split='train'
        )

    voxel = ds[0]['input_occ'].unsqueeze(0).cuda()
    text = ds[0]['text'].unsqueeze(0).cuda()

    rh = RefHead_Text(
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
        )

    ).cuda()

    x_32, x_64, x_128, x_256 = rh(voxel, text)

