import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, shape, dilations=[1, 2, 4, 8]):
        super(ASPP3D, self).__init__()
        self.shape = shape

        # 1x1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(1e-1, True)
        )

        # Atrous convolutions with different dilation rates

        # 3x3x3 convolution branch / dilation=dilations[1]
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                      padding=dilations[1], dilation=dilations[1], bias=True),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(1e-1, True)
        )
        
        # 3x3x3 convolution branch / dilation=dilations[2]
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                      padding=dilations[2], dilation=dilations[2], bias=True),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(1e-1, True)
        )
        
        # 3x3x3 convolution branch / dilation=dilations[2]
        self.branch4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                      padding=dilations[3], dilation=dilations[3], bias=True),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(1e-1, True)
        )
        
        # GlobalAveragePoolng
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((shape[0] // 2, shape[1] // 2, shape[2] // 2)),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(1e-1, True)
        )
        
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)   
        
        # output project
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * (1 + len(dilations)), out_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(1e-1, True)
        )
        

    def forward(self, x):
        
        # Forward through all branches
        x1 = self.branch1(x)  # torch.Size([1, D, X, Y, Z])
        x2 = self.branch2(x)  # torch.Size([1, D, X, Y, Z])
        x3 = self.branch3(x)  # torch.Size([1, D, X, Y, Z])
        x4 = self.branch4(x)  # torch.Size([1, D, X, Y, Z])
        x5 = self.up_scale(self.global_avg_pool(x))  # torch.Size([1, D, X, Y, Z])

        # Cat
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        # Output Project
        x = self.project(x)

        return x



if __name__ == '__main__':
    a = ASPP3D(
        in_channels=64, 
        out_channels=64,
        shape=[256,256,32]
    ).cuda()
    
    v = (1, 64, 256, 256, 32)
    v = torch.randn(v).cuda()

    y = a(v)

    print(y.size())

