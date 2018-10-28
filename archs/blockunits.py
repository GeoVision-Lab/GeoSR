#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-09-26T09:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Shiba-lab
@License: MIT
"""
import numpy as np
import sys
sys.path.append('./result')
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
# ConvBlocks(conv + (bn) + act): without changing size
## ConvBlock: 1 layer conv 
## ConvBlockx2: 2 layer conv
## ConvBlockx3: 3 layer conv
"""
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(ConvBlock, self).__init__()
        if is_bn:
            self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))
        else:
            self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))
    def forward(self, x):
        x = self.block(x)
        return x   

class ConvBlockx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(ConvBlockx2, self).__init__()
        if is_bn:
            self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))
        else:
            self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))                    
    def forward(self, x):
        x = self.block(x)
        return x    
    
class ConvBlockx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(ConvBlockx3, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))

    def forward(self, x):
        x = self.block(x)
        return x   

"""
# DeConvBlocks(deconv + (bn) + act): without changing size
## DeConvBlock: 1 layer conv
## DeConvBlockx2: 2 layer conv
## DeConvBlockx3: 3 layer conv
"""
class DeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(DeConvBlock, self).__init__()
        if is_bn:
            self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
        else:
            self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))
    def forward(self, x):
        x = self.block(x)
        return x 

class DeConvBlockx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(DeConvBlockx2, self).__init__()
        if is_bn:
            self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
        else:
            self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
    def forward(self, x):
        x = self.block(x)
        return x

class DeConvBlockx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(DeConvBlockx3, self).__init__()
        if is_bn:
            self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
        else:
            self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
    def forward(self, x):
        x = self.block(x)
        return x

"""
# UNet decoder blocks
## UNetUpx2: upsampling(nearest/learnable_decon) + ConvBlockx2/DeConvBlockx2
## UNetUpx3: upsampling(nearest/learnable_decon) + ConvBlockx3/DeConvBlockx3
## nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2): size x 2, avoid checkerboard artifacts
"""

class UNetUpx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetUpx2, self).__init__()
        # upsampling and convolution block
        if is_deconv:
            if is_bn:
                self.upscale = nn.Sequential(
                        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                        nn.BatchNorm2d(out_ch),
                        nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
            else:
                self.upscale = nn.Sequential(
                        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                        nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
            self.block = DeConvBlockx2(in_ch, out_ch, is_bn, is_leaky, alpha)
        else:
            if is_bn:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.Upsample(scale_factor=2))
            else:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.Upsample(scale_factor=2))
            self.block = ConvBlockx2(in_ch, out_ch, is_bn, is_leaky, alpha)
    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x1, x2], dim = 1)
        x = self.block(x)
        return x

class UNetUpx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetUpx3, self).__init__()
        # upsampling and convolution block
        if is_deconv:
            if is_bn:
                self.upscale = nn.Sequential(
                        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                        nn.BatchNorm2d(out_ch),
                        nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
            else:
                self.upscale = nn.Sequential(
                        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                        nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
            self.block = DeConvBlockx3(in_ch, out_ch, is_bn, is_leaky, alpha)
        else:
            if is_bn:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.Upsample(scale_factor=2))
            else:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.Upsample(scale_factor=2))
            self.block = ConvBlockx3(in_ch, out_ch, is_bn, is_leaky, alpha)
    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x1, x2], dim = 1)
        x = self.block(x)
        return x


"""
# SegNet decoder blocks
## UNetUpx2: unpooling + ConvBlockx2/DeConvBlockx2
## UNetUpx3: unpooling + ConvBlockx3/DeConvBlockx3
"""

class SegNetUpx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(SegNetUpx2, self).__init__()
        # unpooling and convolution block
        if is_deconv:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.block = DeConvBlockx2(in_ch, out_ch, is_bn, is_leaky=False, alpha=0.1)
        else:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.block = ConvBlockx2(in_ch, out_ch, is_bn, is_leaky=False, alpha=0.1)
        
    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
#        print(x.size())
        x = self.block(x)
        return x

class SegNetUpx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(SegNetUpx3, self).__init__()
        # unpooling and convolution block
        if is_deconv:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.block = DeConvBlockx3(in_ch, out_ch, is_bn, is_leaky=False, alpha=0.1)
        else:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.block = ConvBlockx3(in_ch, out_ch, is_bn, is_leaky=False, alpha=0.1)
        
    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
        x = self.block(x)
        return x

"""
# ResNet blocks
"""
                 
def conv3x3bn(in_ch, out_ch, stride=1):
    #3x3 convolution with padding
    #conv + batch normalization
    convbn = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),)
    return convbn


class ResBasicBlock(nn.Module):
    # jump two blocks
    expansion = 1
    # modified from original pytorch implementation
    # see http://pytorch.org/docs/0.2.0/_modules/torchvision/models/resnet.html

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, is_leaky=False):
        super(ResBasicBlock, self).__init__()
        alpha = 0.1
        self.conv1bn = conv3x3bn(in_ch, out_ch, stride)
        self.relu = nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True)
        self.conv2bn = conv3x3bn(out_ch, out_ch)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1bn(x)
        out = self.relu(out)

        out = self.conv2bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBottleneck(nn.Module):
    # jump three blocks
    expansion = 4
    # modified from original pytorch implementation
    # see http://pytorch.org/docs/0.2.0/_modules/torchvision/models/resnet.html
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, is_leaky=False):
        super(ResBottleneck, self).__init__()
        alpha = 0.1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.relu = nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

"""
https://pytorch.org/docs/0.4.0/_modules/torchvision/models/inception.html
# Inception Blocks
## details: https://zhuanlan.zhihu.com/p/30172532
"""
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x
    


if __name__ == "__main__":
    # Hyper Parameters
    img_col, img_row, nb_channel = 224, 224, 3
    base_kernel = 64
    out_ch = 100
    x = torch.FloatTensor(
            np.random.random((1, nb_channel, img_col, img_row,)))

    generator = ConvBlock(nb_channel, out_ch, True, True)
    gen_y = generator(x)
    print("ConvBlock->:")
    print(" Block output ", gen_y.shape)

    generator = ConvBlockx2(nb_channel, out_ch, True, True)
    gen_y = generator(x)
    print("ConvBlockx2->:")
    print(" Block output ", gen_y.shape)

    generator = ConvBlockx3(nb_channel, out_ch, True, True)
    gen_y = generator(x)
    print("ConvBlockx3->:")
    print(" Block output ", gen_y.shape)
    
    generator = DeConvBlock(nb_channel, out_ch, True, True)
    gen_y = generator(x)
    print("DeConvBlock->:")
    print(" Block output ", gen_y.shape)

    generator = DeConvBlockx2(nb_channel, out_ch, True, True)
    gen_y = generator(x)
    print("DeConvBlockx2->:")
    print(" Block output ", gen_y.shape)

    generator = DeConvBlockx3(nb_channel, out_ch, True, True)
    gen_y = generator(x)
    print("DeConvBlockx3->:")
    print(" Block output ", gen_y.shape)
    