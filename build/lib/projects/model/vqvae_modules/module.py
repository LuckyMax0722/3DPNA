import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn, einsum


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,padding=(0, 1, 1), bias=False)

def conv1x1x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, padding=(0, 0, 1), bias=False)

def conv1x3x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride, padding=(0, 1, 0), bias=False)

def conv3x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, padding=(1, 0, 0), bias=False)

def conv3x1x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride, padding=(1, 0, 1), bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride)


class Asymmetric_Residual_Block(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(Asymmetric_Residual_Block, self).__init__()
        self.conv1 = conv1x3x3(in_filters, out_filters)
        self.act1 = nn.LeakyReLU()          
        self.conv1_2 = conv3x1x3(out_filters, out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1x3(in_filters, out_filters)
        self.act2 = nn.LeakyReLU()

        self.conv3 = conv1x3x3(out_filters, out_filters)
        self.act3 = nn.LeakyReLU()

        if in_filters<32 :
            self.GroupNorm = nn.GroupNorm(8, in_filters)
            self.bn0 = nn.GroupNorm(8, out_filters)
            self.bn0_2 = nn.GroupNorm(8, out_filters)
            self.bn1 = nn.GroupNorm(8, out_filters)
            self.bn2 = nn.GroupNorm(8, out_filters)
        else :
            self.GroupNorm = nn.GroupNorm(32, in_filters)
            self.bn0 = nn.GroupNorm(32, out_filters)
            self.bn0_2 = nn.GroupNorm(32, out_filters)
            self.bn1 = nn.GroupNorm(32, out_filters)
            self.bn2 = nn.GroupNorm(32, out_filters)


    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)
        shortcut = self.bn0(shortcut)

        shortcut = self.conv1_2(shortcut)
        shortcut = self.act1_2(shortcut)
        shortcut = self.bn0_2(shortcut)

        resA = self.conv2(x) 
        resA = self.act2(resA)
        resA = self.bn1(resA)

        resA = self.conv3(resA) 
        resA = self.act3(resA)
        resA = self.bn2(resA)
        resA += shortcut

        return resA


class DownBlock(nn.Module):
    def __init__(self, in_filters, out_filters, pooling=True, drop_out=True, height_pooling=False):
        super(DownBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.residual_block = Asymmetric_Residual_Block(in_filters, out_filters)
        if pooling:
            if height_pooling:
                self.pool = nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2,padding=1, bias=False)
            else:
                self.pool = nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),padding=1, bias=False)

    def forward(self, x):
        resA = self.residual_block(x)
        if self.pooling:
            resB = self.pool(resA) 
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, height_pooling):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3x3(in_filters, out_filters)
        self.trans_act = nn.LeakyReLU()

        self.conv1 = conv1x3x3(out_filters, out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv2 = conv3x1x3(out_filters, out_filters)
        self.act2 = nn.LeakyReLU()

        self.conv3 = conv3x3x3(out_filters, out_filters)
        self.act3 = nn.LeakyReLU()

        if out_filters<32 :
            self.trans_bn = nn.GroupNorm(8, out_filters)
            self.bn1 = nn.GroupNorm(8, out_filters)
            self.bn2 = nn.GroupNorm(8, out_filters)
            self.bn3 = nn.GroupNorm(8, out_filters)
        else :
            self.trans_bn = nn.GroupNorm(32, out_filters)
            self.bn1 = nn.GroupNorm(32, out_filters)
            self.bn2 = nn.GroupNorm(32, out_filters)
            self.bn3 = nn.GroupNorm(32, out_filters)
        
        if height_pooling :
            self.up_subm = nn.ConvTranspose3d(out_filters, out_filters, kernel_size=3, bias=False, stride=2, padding=1, output_padding=1, dilation=1)
        else : 
            self.up_subm = nn.ConvTranspose3d(out_filters, out_filters, kernel_size=(3,3,1), bias=False, stride=(2,2,1), padding=(1,1,0), output_padding=(1,1,0), dilation=1)


    def forward(self, x, skip=False): 
        if skip :
            x, residual = x
        upA = self.trans_dilao(x)
        upA = self.trans_act(upA)
        upA = self.trans_bn(upA) 

        upA = self.up_subm(upA)
        if skip :
            upA += residual
        upE = self.conv1(upA)
        upE = self.act1(upE)
        upE = self.bn1(upE)

        upE = self.conv2(upE)
        upE = self.act2(upE)
        upE = self.bn2(upE)

        upE = self.conv3(upE)
        upE = self.act3(upE)
        upE = self.bn3(upE)
        return upE


class DDCM(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1):
        super(DDCM, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters)
        self.act1_3 = nn.Sigmoid()

        if in_filters<32 :
            self.bn0 = nn.GroupNorm(8, out_filters)
            self.bn0_2 = nn.GroupNorm(8, out_filters)
            self.bn0_3 = nn.GroupNorm(8, out_filters)
        else :
            self.bn0 = nn.GroupNorm(32, out_filters)
            self.bn0_2 = nn.GroupNorm(32, out_filters)
            self.bn0_3 = nn.GroupNorm(32, out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.bn0(shortcut)
        shortcut = self.act1(shortcut)

        shortcut2 = self.conv1_2(x)
        shortcut2 = self.bn0_2(shortcut2)
        shortcut2 = self.act1_2(shortcut2)

        shortcut3 = self.conv1_3(x)
        shortcut3 = self.bn0_3(shortcut3)
        shortcut3 = self.act1_3(shortcut3)
        shortcut = shortcut + shortcut2 + shortcut3

        shortcut = shortcut * x
        return shortcut

def l2norm(t):
    return F.normalize(t, dim = -1)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        self.to_qkv = conv1x1(dim, dim*3, stride=1)
        self.to_out = conv1x1(dim, dim, stride=1)

    def forward(self, x):
        b, c, h, w, Z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h c (x y z)', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = Z)
        return self.to_out(out)

class C_Encoder(nn.Module):
    def __init__(self,  nclasses=20, init_size=16, l_size='882', attention=True):
        super(C_Encoder, self).__init__()
        self.nclasses = nclasses
        self.l_size = l_size
        self.attention = attention

        self.embedding = nn.Embedding(nclasses, init_size)

        self.A = Asymmetric_Residual_Block(init_size, init_size)

        self.downBlock1 = DownBlock(init_size, 2 * init_size, height_pooling=True)
        self.downBlock2 = DownBlock(2 * init_size, 4 * init_size, height_pooling=True)
        self.downBlock3 = DownBlock(4 * init_size, 8 * init_size, height_pooling=False)
        self.downBlock4 = DownBlock(8 * init_size, 16 * init_size, height_pooling=False)
        
        if self.l_size == '32322':
            self.midBlock1 = Asymmetric_Residual_Block(4 * init_size, 4 * init_size)
            self.attention = Attention(4 * init_size, 32)
            self.midBlock2 = Asymmetric_Residual_Block(4 * init_size, 4 * init_size)
            self.out = nn.Conv3d(4 * init_size, nclasses, kernel_size=3, stride=1, padding=1,bias=True)
        elif self.l_size == '16162':
            self.midBlock1 = Asymmetric_Residual_Block(8 * init_size, 8 * init_size)
            self.attention = Attention(8 * init_size, 32)
            self.midBlock2 = Asymmetric_Residual_Block(8 * init_size, 8 * init_size)
            self.out = nn.Conv3d(8 * init_size, nclasses, kernel_size=3, stride=1, padding=1,bias=True)
        elif self.l_size == '882':
            self.midBlock1 = Asymmetric_Residual_Block(16 * init_size, 16 * init_size)
            self.attention = Attention(16 * init_size, 32)
            self.midBlock2 = Asymmetric_Residual_Block(16 * init_size, 16 * init_size)
            self.out = nn.Conv3d(16 * init_size, nclasses, kernel_size=3, stride=1, padding=1,bias=True)
        
    def forward(self, x, out_conv=True):
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = self.A(x)
        x, down1b = self.downBlock1(x)
        x, down2b = self.downBlock2(x)

        if self.l_size == '882':
            x, down3b = self.downBlock3(x)
            x, down4b = self.downBlock4(x)
        elif self.l_size == '16162':
            x, down3b = self.downBlock3(x)
        
        if self.attention : 
            x = self.midBlock1(x) # (4, 128, 32, 32, 2)
            x = self.attention(x)
            x = self.midBlock2(x) # (4, 128, 32, 32, 2)
        if out_conv : 
            x = self.out(x)

        if self.l_size == '882':
            return x, down4b, down3b, down2b, down1b
        elif self.l_size == '16162':
            return x, down3b, down2b, down1b
        else:
            return x, down2b, down1b

class C_Decoder(nn.Module):
    def __init__(self, nclasses=20, init_size=16, l_size='882', attention=True):
        super(C_Decoder, self).__init__()
        self.nclasses = nclasses
        self.l_size = l_size
        self.attention = attention

        if l_size == '882':
            self.conv_in = nn.Conv3d(nclasses, 16 * init_size, kernel_size=3, stride=1, padding=1,bias=True)
            self.midBlock1 = Asymmetric_Residual_Block(16 * init_size, 16 * init_size)
            self.attention = Attention(16 * init_size, 32)
            self.midBlock2 = Asymmetric_Residual_Block(16 * init_size, 16 * init_size)
        elif l_size == '16162':
            self.conv_in = nn.Conv3d(nclasses, 8 * init_size, kernel_size=3, stride=1, padding=1,bias=True)
            self.midBlock1 = Asymmetric_Residual_Block(8 * init_size, 8 * init_size)
            self.attention = Attention(8 * init_size, 32)
            self.midBlock2 = Asymmetric_Residual_Block(8 * init_size, 8 * init_size)
        elif (l_size =='32322'):
            self.conv_in = nn.Conv3d(nclasses, 4 * init_size, kernel_size=3, stride=1, padding=1,bias=True)
            self.midBlock1 = Asymmetric_Residual_Block(4 * init_size, 4 * init_size)
            self.attention = Attention(4 * init_size, 32)
            self.midBlock2 = Asymmetric_Residual_Block(4 * init_size, 4 * init_size)

        self.upBlock4 = UpBlock(16 * init_size, 8 * init_size, height_pooling=False)
        self.upBlock3 = UpBlock(8 * init_size, 4 * init_size, height_pooling=False)
        self.upBlock2 = UpBlock(4 * init_size, 2 * init_size, height_pooling=True)
        self.upBlock1 = UpBlock(2 * init_size, init_size, height_pooling=True)
        self.DDCM = DDCM(init_size, init_size)

    def forward(self, x, in_conv=True):
        if in_conv :
            x = self.conv_in(x)

        if self.attention : 
            x = self.midBlock1(x)
            x = self.attention(x)
            x = self.midBlock2(x)                    

        if self.l_size == '882':
            x4 = self.upBlock4(x)  # x4: torch.Size([1, 256, 32, 32, 8])
            x3 = self.upBlock3(x4)  # x3: torch.Size([1, 128, 64, 64, 8])
            
        elif self.l_size == '16162':
            x3 = self.upBlock3(x)

        x2 = self.upBlock2(x3)  # x2: torch.Size([1, 64, 128, 128, 16]
        up1 = self.upBlock1(x2)  # up1: torch.Size([1, 32, 128, 128, 16]

        up0 = self.DDCM(up1) # up0: torch.Size([1, 32, 128, 128, 16]
        x1 = torch.cat((up1, up0), 1)  # x1: torch.Size([1, 64, 256, 256, 32])

        #logits = self.logits(up) 
        #return logits

        return x1, x2, x3