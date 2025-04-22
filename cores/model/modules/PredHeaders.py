import torch
import torch.nn as nn

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

        x = self.conv_head(x)
   
        return x

class PredHeaders(nn.Module):
    def __init__(
        self, 
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