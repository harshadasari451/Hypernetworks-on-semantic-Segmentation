""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

class DropOut(nn.Module):
    def __init__(self, drop_rate,drop_train):
        super().__init__()
        self.drop_rate = drop_rate
        self.drop_train = drop_train
    
    def forward(self, x):
        return F.dropout(x,p=self.drop_rate,training=self.drop_train)
    

class HashConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, period, 
                 stride=1, padding=0, bias=True,
                 key_pick='hash', learn_key=True):
        super(HashConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        w = torch.zeros(self.out_channels, self.in_channels, *self.kernel_size)
        # print(f'w shape {w.shape}')
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        self.w = nn.Parameter(w)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    
        o_dim = self.in_channels*self.kernel_size[0]*self.kernel_size[1]
        # TODO(briancheung): The line below will cause problems when saving a model
        o = torch.from_numpy( np.random.binomial( p=.5, n=1, size = (o_dim, period) ).astype(np.float32) * 2 - 1 )
        self.o = nn.Parameter(o, requires_grad=True)

    def forward(self, x, time):
        net_time = time % self.o.shape[1]
        o = self.o[:, net_time].view(1,
                                     self.in_channels,
                                     self.kernel_size[0],
                                     self.kernel_size[1])
        return F.conv2d(x, self.w*o, self.bias, stride=self.stride, padding=self.padding)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class HashDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, period, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # self.double_conv = nn.Sequential(
        #     HashConv2d(in_channels, mid_channels, kernel_size=3, period=period, padding=1),
        #     nn.InstanceNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     HashConv2d(mid_channels, out_channels, kernel_size=3, period=period, padding=1),
        #     nn.InstanceNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.hash_conv_2d_1 = HashConv2d(in_channels, mid_channels, kernel_size=3, period=period, padding=1)
        self.inst_norm_2d_1 = nn.InstanceNorm2d(mid_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.hash_conv_2d_2 = HashConv2d(mid_channels, out_channels, kernel_size=3, period=period, padding=1)
        self.inst_norm_2d_2 = nn.InstanceNorm2d(out_channels)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x, time):
        out = self.hash_conv_2d_1(x, time)
        out1 = self.inst_norm_2d_1(out)
        out2 = self.relu_1(out1)
        out3 = self.hash_conv_2d_2(out2, time)
        out4 = self.inst_norm_2d_2(out3)
        out5 = self.relu_2(out4)
        return out5

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, drop_train=True, drop_rate=0.5, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_train = drop_train
        self.drop_rate = drop_rate

        #adding 1X1 conv to input representation
        self.softconv = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.inc = DoubleConv(32, 64)
        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.drop1 = DropOut(drop_rate,drop_train)
        self.down2 = Down(128, 256)
        self.drop2 = DropOut(drop_rate,drop_train)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.drop3 = DropOut(drop_rate,drop_train)
        self.down4 = Down(512, 1024 // factor)
        self.drop4 = DropOut(drop_rate,drop_train)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.drop5 = DropOut(drop_rate,drop_train)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, input_data):
        #F.dropout(x2,p=self.dropout,training=self.drop_train)
        x = self.softconv(input_data)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x22 = self.drop1(x2)
        x3 = self.down2(x22)
        x33 = self.drop2(x3)
        x4 = self.down3(x33)
        x44 = self.drop3(x4)
        x5 = self.down4(x44)
        x55= self.drop4(x5)
        x = self.up1(x55, x4)
        x = self.drop5(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetMedChannels(nn.Module):
    def __init__(self, n_channels, n_classes, drop_train=True, drop_rate=0.5, bilinear=True):
        super(UNetMedChannels, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_train = drop_train
        self.drop_rate = drop_rate

        #adding 1X1 conv to input representation
        self.softconv = nn.Conv2d(in_channels=n_channels, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.inc = DoubleConv(16, 32)
        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(32, 64)
        self.drop1 = DropOut(drop_rate,drop_train)
        self.down2 = Down(64, 128)
        self.drop2 = DropOut(drop_rate,drop_train)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.drop3 = DropOut(drop_rate,drop_train)
        self.down4 = Down(256, 512 // factor)
        self.drop4 = DropOut(drop_rate,drop_train)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.drop5 = DropOut(drop_rate,drop_train)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, input_data):
        #F.dropout(x2,p=self.dropout,training=self.drop_train)
        x = self.softconv(input_data)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x22 = self.drop1(x2)
        x3 = self.down2(x22)
        x33 = self.drop2(x3)
        x4 = self.down3(x33)
        x44 = self.drop3(x4)
        x5 = self.down4(x44)
        x55= self.drop4(x5)
        x = self.up1(x55, x4)
        x = self.drop5(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetSmallChannels(nn.Module):
    def __init__(self, n_channels, n_classes, drop_train=True, drop_rate=0.5, bilinear=True):
        super(UNetSmallChannels, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_train = drop_train
        self.drop_rate = drop_rate

        #adding 1X1 conv to input representation
        self.softconv = nn.Conv2d(in_channels=n_channels, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.inc = DoubleConv(8, 16)
        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(16, 32)
        self.drop1 = DropOut(drop_rate,drop_train)
        self.down2 = Down(32, 64)
        self.drop2 = DropOut(drop_rate,drop_train)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.drop3 = DropOut(drop_rate,drop_train)
        self.down4 = Down(128, 256 // factor)
        self.drop4 = DropOut(drop_rate,drop_train)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.drop5 = DropOut(drop_rate,drop_train)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, input_data):
        #F.dropout(x2,p=self.dropout,training=self.drop_train)
        x = self.softconv(input_data)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x22 = self.drop1(x2)
        x3 = self.down2(x22)
        x33 = self.drop2(x3)
        x4 = self.down3(x33)
        x44 = self.drop3(x4)
        x5 = self.down4(x44)
        x55= self.drop4(x5)
        x = self.up1(x55, x4)
        x = self.drop5(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetTinyChannels(nn.Module):
    def __init__(self, n_channels, n_classes, drop_train=True, drop_rate=0.5, bilinear=True):
        super(UNetTinyChannels, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_train = drop_train
        self.drop_rate = drop_rate

        #adding 1X1 conv to input representation
        self.softconv = nn.Conv2d(in_channels=n_channels, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.inc = DoubleConv(4, 8)
        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(8, 16)
        self.drop1 = DropOut(drop_rate,drop_train)
        self.down2 = Down(16, 32)
        self.drop2 = DropOut(drop_rate,drop_train)
        self.down3 = Down(32, 64)
        factor = 2 if bilinear else 1
        self.drop3 = DropOut(drop_rate,drop_train)
        self.down4 = Down(64, 128 // factor)
        self.drop4 = DropOut(drop_rate,drop_train)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.drop5 = DropOut(drop_rate,drop_train)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, input_data):
        #F.dropout(x2,p=self.dropout,training=self.drop_train)
        x = self.softconv(input_data)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x22 = self.drop1(x2)
        x3 = self.down2(x22)
        x33 = self.drop2(x3)
        x4 = self.down3(x33)
        x44 = self.drop3(x4)
        x5 = self.down4(x44)
        x55= self.drop4(x5)
        x = self.up1(x55, x4)
        x = self.drop5(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetTinyThreeBlock(nn.Module):
    def __init__(self, n_channels, n_classes, drop_train=True, drop_rate=0.5, bilinear=True):
        super(UNetTinyThreeBlock, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_train = drop_train
        self.drop_rate = drop_rate

        #adding 1X1 conv to input representation
        self.softconv = nn.Conv2d(in_channels=n_channels, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.inc = DoubleConv(4, 8)
        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(8, 16)
        self.drop1 = DropOut(drop_rate,drop_train)
        self.down2 = Down(16, 32)
        self.drop2 = DropOut(drop_rate,drop_train)
        factor = 2 if bilinear else 1
        self.down3 = Down(32, 64 // factor)
        self.drop3 = DropOut(drop_rate,drop_train)
        #self.down4 = Down(64, 128 // factor)
        #self.drop4 = DropOut(drop_rate,drop_train)
        #self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.drop5 = DropOut(drop_rate,drop_train)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, input_data):
        #F.dropout(x2,p=self.dropout,training=self.drop_train)
        x = self.softconv(input_data)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x22 = self.drop1(x2)
        x3 = self.down2(x22)
        x33 = self.drop2(x3)
        x4 = self.down3(x33)
        x44 = self.drop3(x4)
        #x5 = self.down4(x44)
        #x55 = self.drop4(x5)
        #x = self.up1(x55, x4)
        x = self.up2(x44, x3)
        x = self.drop5(x)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetTinyTwoBlock(nn.Module):
    def __init__(self, n_channels, n_classes, drop_train=True, drop_rate=0.5, bilinear=True):
        super(UNetTinyTwoBlock, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_train = drop_train
        self.drop_rate = drop_rate

        #adding 1X1 conv to input representation
        self.softconv = nn.Conv2d(in_channels=n_channels, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.inc = DoubleConv(4, 8)
        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(8, 16)
        self.drop1 = DropOut(drop_rate,drop_train)
        factor = 2 if bilinear else 1
        self.down2 = Down(16, 32 // factor)
        self.drop2 = DropOut(drop_rate,drop_train)
        #self.down3 = Down(32, 64 // factor)
        #self.drop3 = DropOut(drop_rate,drop_train)
        #self.down4 = Down(64, 128 // factor)
        #self.drop4 = DropOut(drop_rate,drop_train)
        #self.up1 = Up(128, 64 // factor, bilinear)
        #self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.drop5 = DropOut(drop_rate,drop_train)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, input_data):
        #F.dropout(x2,p=self.dropout,training=self.drop_train)
        x = self.softconv(input_data)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x22 = self.drop1(x2)
        x3 = self.down2(x22)
        x33 = self.drop2(x3)
        #x4 = self.down3(x33)
        #x44 = self.drop3(x4)
        #x5 = self.down4(x44)
        #x55 = self.drop4(x5)
        #x = self.up1(x55, x4)
        #x = self.up2(x44, x3)
        x = self.up3(x33, x2)
        x = self.drop5(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class HashUNetTinyTwoBlock(nn.Module):
    def __init__(self, n_channels, n_classes, drop_train=True, drop_rate=0.5, bilinear=True):
        super(HashUNetTinyTwoBlock, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_train = drop_train
        self.drop_rate = drop_rate

        #adding 1X1 conv to input representation
        self.softconv = HashConv2d(in_channels=n_channels, out_channels=4, kernel_size=1, period=2, stride=1, padding=0)
        self.inc = HashDoubleConv(4, 8, 2)
        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(8, 16)
        self.drop1 = DropOut(drop_rate,drop_train)
        factor = 2 if bilinear else 1
        self.down2 = Down(16, 32 // factor)
        self.drop2 = DropOut(drop_rate,drop_train)
        #self.down3 = Down(32, 64 // factor)
        #self.drop3 = DropOut(drop_rate,drop_train)
        #self.down4 = Down(64, 128 // factor)
        #self.drop4 = DropOut(drop_rate,drop_train)
        #self.up1 = Up(128, 64 // factor, bilinear)
        #self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.drop5 = DropOut(drop_rate,drop_train)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, input_data, time):
        x = self.softconv(input_data, time)
        x1 = self.inc(x, time)
        x2 = self.down1(x1)
        x22 = self.drop1(x2)
        x3 = self.down2(x22)
        x33 = self.drop2(x3)
        x = self.up3(x33, x2)
        x = self.drop5(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # return logits
        return logits, x.detach().clone(), x1.detach().clone()