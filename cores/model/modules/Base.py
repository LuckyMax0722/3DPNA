import torch
import torch.nn as nn

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
        geo_feat_channels
    ):

        super().__init__()
        
        self.convblock = ConvBlock(input_channels=geo_feat_channels * 2, output_channels=geo_feat_channels)

    def forward(self, skip, x):

        x = torch.cat([skip, x], dim=1)
        x = self.convblock(x)

        return x
