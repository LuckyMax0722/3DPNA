import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, z_down, padding_mode, kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        self.z_down = z_down
        self.conv0 = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode)
        self.convblock1 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
        if self.z_down :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        else :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        self.convblock2 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )

    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        x = self.conv0(x)  # [b, geo_feat_channels, X, Y, Z]

        residual_feat = x
        x = self.convblock1(x)  # [b, geo_feat_channels, X, Y, Z]
        x = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
        x = self.downsample(x)  # [b, geo_feat_channels, X//2, Y//2, Z//2]

        residual_feat = x
        x = self.convblock2(x)
        x = x + residual_feat

        return x  # [b, geo_feat_channels, X//2, Y//2, Z//2]

class TPVEncoder(nn.Module):
    def __init__(
        self, 
        num_class,
        geo_feat_channels,
        z_down,
        padding_mode,
        ) -> None:

        super().__init__()

        self.embedding = nn.Embedding(num_class, geo_feat_channels)

        self.geo_encoder = Encoder(geo_feat_channels, z_down, padding_mode)

        self.norm = nn.InstanceNorm2d(geo_feat_channels)

        self.geo_feat_dim = geo_feat_channels


    def encode(self, vol):
        x = vol.detach().clone()
        x[x == 255] = 0
            
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)
        vol_feat = self.geo_encoder(x)

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)
        
        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]
 
    
    def forward(self, vol):
        '''
        Output:
            feat_map: [
                torch.Size([1, 32, 128, 128])
                torch.Size([1, 32, 128, 32])
                torch.Size([1, 32, 128, 32])
            ]
        '''
        feat_map = self.encode(vol)
        return feat_map


if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/tpv/AutoEncoderGroupSkip.py
    import numpy as np

    x = np.load('/u/home/caoh/datasets/SemanticKITTI/dataset/labels/00/000000_1_1.npy')
    x = torch.from_numpy(x).long().cuda().unsqueeze(0)
    

    voxel_input = np.load('/u/home/caoh/datasets/SemanticKITTI/dataset/pred/MonoScene/00/000000.npy')
    voxel_input = torch.from_numpy(voxel_input).long().cuda().unsqueeze(0)

    AEGS = TPVEncoder(
        num_class=20,
        geo_feat_channels=32,
        z_down=False,
        padding_mode='replicate',
    ).cuda()

    feat = AEGS(x)

    for i in range(3):
        print(feat[i].size())