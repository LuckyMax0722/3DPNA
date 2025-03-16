import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from projects.loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.model.VQVQE import vqvae

class Header(nn.Module): # mlp as perdition head
    def __init__(
        self,
        geo_feat_channels,
        class_num,
    ):
        super(Header, self).__init__()
        self.geo_feat_channels = geo_feat_channels
        self.class_num = class_num
        
        self.conv_head = nn.Sequential(
            nn.Conv3d(self.geo_feat_channels, self.class_num, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        # [1, 64, 256, 256, 32]
        res = {} 
            
        ssc_logit = self.conv_head(x)
        
        res["ssc_logit"] = ssc_logit
        
        return res

class RefHead_VQ(nn.Module):
    def __init__(
        self,
        num_class,
        init_size,
        l_size,
        l_attention,
        vq_size,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
        empty_idx=0,
    ):
        super(RefHead_VQ, self).__init__()
        
        self.empty_idx = empty_idx

        self.vqvae = vqvae(
            num_classes = num_class,
            init_size = init_size,
            l_size = l_size,
            l_attention = l_attention,
            vq_size = vq_size
        )

        self.pred_head_4 = Header(4 * init_size, num_class)
        self.pred_head_2 = Header(2 * init_size, num_class)
        self.pred_head_1 = Header(2 * init_size, num_class)
        
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

        x1, x2, x3, vq_loss = self.vqvae(x)
        
        x3 = self.pred_head_4(x3)
        
        x2 = self.pred_head_2(x2)
        
        x1 = self.pred_head_1(x1)

        return x1, x2, x3, vq_loss
    
    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [1, 2, 4]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict
    
    def loss_vq(self, vq_loss):
        loss_dict = {}

        loss_dict['loss_vq'] = vq_loss

        return loss_dict


if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_vq.py
    v = RefHead_VQ(
        num_class = 20,
        init_size = 32,
        l_size = '882',
        l_attention = True,
        vq_size = 50,
        balance_cls_weight=False
    ).cuda()

    x = np.load('/u/home/caoh/datasets/SemanticKITTI/dataset/pred/MonoScene/00/000000.npy')


    x = torch.from_numpy(x).long().cuda().unsqueeze(0)

    
    v(x)