import torch
import torch.nn as nn 
import torch.nn.functional as F

import pytorch_lightning as pl

from einops import rearrange

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from configs.config import CONF

from projects.model.tpv.TPVAE_V2 import TPVGenerator, TPVAggregator
from projects.model.tpv.blocks import Encoder, Header, Decoder

from projects.loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss

def get_image(img_path):
    crop_size = 480
    padding = [0.0] * 3
    image = Image.open(img_path)
    image = np.array(image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0)

    return image


class RefHead_TPV_Lseg_V3(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,
        img_feat_channels,
        kv_dim,
        z_down,

        dim_head,
        heads,
        ffn_cfg,
    
        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
        padding_mode='replicate',
    ):
        super(RefHead_TPV_Lseg_V3, self).__init__()

        if z_down:
            shape_256 = [256, 256, 32]
            shape_128 = (128, 128, 16)
            shape_64 = (64, 64, 8)
            shape_32 = (32, 32, 4)
        else:
            raise ValueError("Wrong Shape Size.")

        self.v_embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, C, H, W]

        self.conv_in = nn.Conv3d(
            geo_feat_channels, 
            geo_feat_channels, 
            kernel_size=(5, 5, 3), 
            stride=(1, 1, 1), 
            padding=(2, 2, 1), 
            bias=True, 
            padding_mode=padding_mode
        )

        self.geo_encoder_128 = Encoder(
            geo_feat_channels=geo_feat_channels,
            z_down=z_down, 
            padding_mode=padding_mode
        )


        self.tpv = TPVGenerator(
            embed_dims=geo_feat_channels,
            split=[8, 8, 8],
            grid_size=shape_256,
            pooler='avg',
        )

        self.fuser = TPVAggregator(
            embed_dims=geo_feat_channels,
        )

        self.decoder_256 = Decoder(
            geo_feat_channels=geo_feat_channels
        )

        self.pred_head_256 = Header(geo_feat_channels, num_class)
        self.pred_head_128 = Header(geo_feat_channels, num_class)


        # voxel losses
        self.empty_idx = empty_idx
        
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

    def forward(self, voxel, image, image_seg):
        x = voxel.detach().clone()
        x[x == 255] = 0
            
        x = self.v_embedding(x)
        x = x.permute(0, 4, 1, 2, 3)

        x_256 = self.conv_in(x)

        #x_128 = self.geo_encoder_128(x_256)
        #tpv_feat, _ = self.tpv(x_128)

        tpv_feat, _ = self.tpv(x_256)

        yz = tpv_feat[2]
        

        # img

        print(image_seg.size())
        #tpv_feat_128 = self.sgm_128(tpv_feat, seg_feat[1])

        x_128, _ = self.fuser(tpv_feat_128, x_128)

        x_256 = self.decoder_256(x_256, x_128)

        x_256 = self.pred_head_256(x_256)
        x_128 = self.pred_head_128(x_128)

        return x_128, x_256

    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict

def contrastive_loss_with_pseudo_masks(X, S, num_classes=20, tau=0.1):
    # ----------------------------------------------------
    # 1) 下采样伪语义分割掩码 S 使其与 X 的 spatial 分辨率对齐
    # ----------------------------------------------------
    # X.shape = [1, C, Y, Z] = [1, 32, 256, 32]
    # S.shape = [1, 1, 384, 1280]
    # 我们用最近邻插值 (nearest) 的方式下采样到 (H=32, W=256)

    X = rearrange(X, 'b c y z -> b c z y')
    # X.shape = [1, C, H, W] = [1, 32, 32, 256]

    B, Cx, H, W = X.shape  # B=1, Cx=32, H=32, W=256
    S = F.interpolate(
        S.float(), 
        size=(H, W), 
        mode='nearest'
    ).long()  # 变为 [1, 1, 32, 256]
    
    # ----------------------------------------------------
    # 2) 计算每个类的原型向量 (prototype)
    #    这里的做法：对属于同一类的像素特征取平均
    # ----------------------------------------------------
    # 先把特征图展平为 [1, Cx, H*W]
    X_flat = X.view(B, Cx, -1)  # [1, 32, 32*256] = [1, 32, 8192]
    # 把掩码也展平为 [1, H*W]
    S_flat = S.view(B, -1) # [1, 32*256] = [1, 8192]

    # 收集所有类别的原型向量
    prototypes = []
    for cls_id in range(num_classes):
        # 找到属于该类的所有像素位置
        mask_cls = (S_flat == cls_id)  # [1, 8192]，bool
        # 计算这些像素在特征图中的平均特征
        if mask_cls.sum() > 0:
            # 只收集属于该类的位置
            # X_flat[:, :, mask_cls] -> [1, 32, (该类像素数)]
            feat_cls = X_flat[:, :, mask_cls]
            # 先在最后一个维度 (像素数) 做平均 -> 得到 [1, 32]
            proto_cls = feat_cls.mean(dim=2)  # [1, 32]
        else:
            # 如果某类在该图中不存在像素，可以置为零向量或随机向量
            proto_cls = torch.zeros((1, Cx), device=X.device)

        prototypes.append(proto_cls)

    # 拼接成 [num_classes, Cx]
    prototypes = torch.cat(prototypes, dim=0)  # [20, 32]

    # ----------------------------------------------------
    # 3) 对每个像素的特征与所有类原型做相似度 (点积)，得到 logits
    # ----------------------------------------------------
    # prototypes: [num_classes, Cx] = [20, 32]
    # X_flat:     [B, Cx, H*W]      = [1, 32, 8192]
    # 点积后：logits = [B, num_classes, H*W]
    # 先把 prototypes 视作 [1, num_classes, Cx] 再与 X_flat 做 batch 矩阵乘
    prototypes_expanded = prototypes.unsqueeze(0)  # [1, 20, 32]
    # matmul: (1, 20, 32) x (1, 32, 8192) -> (1, 20, 8192)
    logits = torch.matmul(prototypes_expanded, X_flat) / tau  # 除以温度系数
    

    # ----------------------------------------------------
    # 4) 使用交叉熵来计算对比损失
    #    - logits 形状: [1, num_classes, H*W]
    #    - target 形状: [1, H*W]
    #    PyTorch 的 F.cross_entropy 需要 [N, C, ...] 的 logits，
    #    以及 [N, ...] 的 target，并在 C 维度上做 softmax。
    # ----------------------------------------------------
    # 这里 B=1，所以可以直接把 [1, 20, 8192] -> [1, 8192, 20]，target -> [1, 8192]
    # 再在 batch 维度上进行 cross_entropy。
    logits = logits.permute(0, 2, 1)  # -> [1, 8192, 20]
    target = S_flat  # [1, 8192]

    # 在只有 batch=1 的情况下，我们可以直接对 logits[0], target[0] 做损失
    # 若需要支持 batch>1，则需做一些 reshape/flatten 再一次性计算。
    loss = F.cross_entropy(logits[0], target[0], reduction='mean')

    return loss


if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_tpv_lseg_v3.py
    
    from projects.datasets import SemanticKITTIDataModule, SemanticKITTIDataset

    ds = SemanticKITTIDataset(
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model='CGFormer',
         vlm_model = 'Lseg',
        split='train'
        )

    image = ds[0]['img'].unsqueeze(0).cuda()
    image_seg = ds[0]['img_seg'].unsqueeze(0).cuda()
    voxel = ds[0]['input_occ'].unsqueeze(0).cuda()

    rh = RefHead_TPV_Lseg_V3(
        num_class=20,
        geo_feat_channels=32,
        img_feat_channels=2048,
        kv_dim=256,
        z_down=True,
        dim_head=8,
        heads=4,
        ffn_cfg=dict(
            type='FFN',
            embed_dims=32,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.1,
            add_identity=True
        ),

        balance_cls_weight=True,
        class_frequencies=[
            5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
            8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
            4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
            1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
        ]

    ).cuda()

    rh(voxel, image, image_seg)
    
