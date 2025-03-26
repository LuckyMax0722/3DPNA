import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, z_down, padding_mode, kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        self.z_down = z_down
        
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
        residual_feat = x
        x = self.convblock1(x)  # [b, geo_feat_channels, X, Y, Z]
        x = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]

        x = self.downsample(x)  # [b, geo_feat_channels, X//2, Y//2, Z//2]

        residual_feat = x
        x = self.convblock2(x)
        x = x + residual_feat

        return x  # [b, geo_feat_channels, X//2, Y//2, Z//2]

class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class DecoderMLPSkipConcat(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers, posenc=0) -> None:
        super().__init__()
        self.posenc = posenc
        if posenc > 0:
            self.PE = SinusoidalEncoder(in_channels, 0, posenc, use_identity=True)
            in_channels = self.PE.latent_dim
        first_layer_list = [nn.Linear(in_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2):
            first_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            first_layer_list.append(nn.ReLU())
        self.first_layers = nn.Sequential(*first_layer_list)
        
        second_layer_list = [nn.Linear(in_channels + hidden_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2 - 1):
            second_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            second_layer_list.append(nn.ReLU())
        second_layer_list.append(nn.Linear(hidden_channels, out_channels))
        self.second_layers = nn.Sequential(*second_layer_list)
    
    def forward(self, x):
        if self.posenc > 0:
            x = self.PE(x)
        h = self.first_layers(x)
        h = torch.cat([x, h], dim=-1)
        h = self.second_layers(h)
        return h
        
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def compose_triplane_channelwise(feat_maps):
    h_xy, h_xz, h_yz = feat_maps # (H, W), (H, D), (W, D)
    assert h_xy.shape[1] == h_xz.shape[1] == h_yz.shape[1]
    C, H, W = h_xy.shape[-3:]
    D = h_xz.shape[-1]

    newH = max(H, W)
    newW = max(W, D)
    h_xy = F.pad(h_xy, (0, newW - W, 0, newH - H))
    h_xz = F.pad(h_xz, (0, newW - D, 0, newH - H))
    h_yz = F.pad(h_yz, (0, newW - D, 0, newH - W))
    h = torch.cat([h_xy, h_xz, h_yz], dim=1) # (B, 3C, H, W)

    return h, (H, W, D)

def decompose_triplane_channelwise(composed_map, sizes):
    H, W, D = sizes
    C = composed_map.shape[1] // 3
    h_xy = composed_map[:, :C, :H, :W]
    h_xz = composed_map[:, C:2*C, :H, :D]
    h_yz = composed_map[:, 2*C:, :W, :D]
    return h_xy, h_xz, h_yz

class TriplaneGroupResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False, ks=3, input_norm=True, input_act=True):
        super().__init__()
        in_channels *= 3
        out_channels *= 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        
        self.input_norm = input_norm
        if input_norm and input_act:
            self.in_layers = nn.Sequential(
                # nn.GroupNorm(num_groups=3, num_channels=in_channels, eps=1e-6, affine=True),
                SiLU(),
                nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            )
        elif not input_norm:
            if input_act:
                self.in_layers = nn.Sequential(
                    SiLU(),
                    nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
            else:
                self.in_layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
        else:
            raise NotImplementedError

        self.norm_xy = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)
        self.norm_xz = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)
        self.norm_yz = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)

        self.out_layers = nn.Sequential(
            # nn.GroupNorm(num_groups=3, num_channels=out_channels, eps=1e-6, affine=True),
            SiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            ),
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, feat_maps):
        if self.input_norm:
            feat_maps = [self.norm_xy(feat_maps[0]), self.norm_xz(feat_maps[1]), self.norm_yz(feat_maps[2])]
        x, (H, W, D) = compose_triplane_channelwise(feat_maps)

        if self.up:
            raise NotImplementedError
        else:
            h = self.in_layers(x)
        
        h_xy, h_xz, h_yz = decompose_triplane_channelwise(h, (H, W, D))
        h_xy = self.norm_xy(h_xy)
        h_xz = self.norm_xz(h_xz)
        h_yz = self.norm_yz(h_yz)
        h, _ = compose_triplane_channelwise([h_xy, h_xz, h_yz])

        h = self.out_layers(h)
        h = h + self.shortcut(x)
        h_maps = decompose_triplane_channelwise(h, (H, W, D))
        return h_maps

class TriplaneUpsample2x(nn.Module):
    def __init__(self, tri_z_down, conv_up, channels=None) -> None:
        super().__init__()
        self.tri_z_down = tri_z_down
        self.conv_up = conv_up
        if conv_up :
            if self.tri_z_down:
                self.conv_xy = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
                self.conv_xz = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
                self.conv_yz = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
            else :
                self.conv_xy = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
                self.conv_xz = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=(1,0), stride=(2, 1))
                self.conv_yz = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=(1,0), stride=(2, 1))

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]
        if self.conv_up:
            #tpl_xy = self.conv_xy(tpl_xy).unsqueeze(-1)
            #tpl_xz = self.conv_xz(tpl_xz).unsqueeze(2)
            #tpl_yz = self.conv_yz(tpl_yz).unsqueeze(3)
            tpl_xy = self.conv_xy(tpl_xy)
            tpl_xz = self.conv_xz(tpl_xz)
            tpl_yz = self.conv_yz(tpl_yz)
        else : 
            tpl_xy = F.interpolate(tpl_xy, scale_factor=2, mode='bilinear', align_corners=False)
            if self.tri_z_down:
                tpl_xz = F.interpolate(tpl_xz, scale_factor=2, mode='bilinear', align_corners=False)
                tpl_yz = F.interpolate(tpl_yz, scale_factor=2, mode='bilinear', align_corners=False)
            else :    
                tpl_xz = F.interpolate(tpl_xz, scale_factor=(2, 1), mode='bilinear', align_corners=False)
                tpl_yz = F.interpolate(tpl_yz, scale_factor=(2, 1), mode='bilinear', align_corners=False)
                
        return [tpl_xy, tpl_xz, tpl_yz]

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
        ssc_logit = self.conv_head(x)
        
        return ssc_logit

class ConvBlock(nn.Module):
    def __init__(
        self, 
        input_channels, 
        output_channels, 
        padding_mode='replicate', 
        stride=(1, 1, 1), 
        kernel_size = (5, 5, 3), 
        padding = (2, 2, 1)):
        
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
    def __init__(self, 
                 geo_feat_channels, 
                 padding_mode='replicate', 
                 stride=(1, 1, 1), 
                 kernel_size = (5, 5, 3), 
                 padding = (2, 2, 1)):
        super().__init__()

        self.convblock = ConvBlock(input_channels=geo_feat_channels * 2, output_channels=geo_feat_channels)

            
    def forward(self, skip, x):

        x = torch.cat([x, skip], dim=1)
        x = self.convblock(x)

        return x


class Decoder(nn.Module):
    def __init__(self, geo_feat_channels):
        super().__init__()
        self.convblock = ResConvBlock(geo_feat_channels=geo_feat_channels)
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, skip, x):
        x = self.up_scale(x)
        x = self.convblock(skip, x)
        
        return x