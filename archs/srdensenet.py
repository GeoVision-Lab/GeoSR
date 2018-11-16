#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T18:36:20+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Shiba-lab
@License: MIT
"""
import math
import numpy as np
import torch
import torch.nn as nn


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class _Dense_Block(nn.Module):
    def __init__(self, in_channel, base_kernel):
        super(_Dense_Block, self).__init__()
        self.base_kernel = base_kernel//8
        kernels = [int(x * self.base_kernel) for x in [1, 2, 3, 4, 5, 6, 7]]
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channel, kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(kernels[0], kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(kernels[1], kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(kernels[2], kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(kernels[3], kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(kernels[4], kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(kernels[5], kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(kernels[6], kernels[0], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))
        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1))
        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))
        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))
        conv6 = self.relu(self.conv6(cout5_dense))
        cout6_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6], 1))
        conv7 = self.relu(self.conv7(cout6_dense))
        cout7_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7], 1))
        conv8 = self.relu(self.conv8(cout7_dense))
        cout8_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8], 1))
        return cout8_dense

class SRDenseNet(nn.Module):
    """
    Superresolution using DenseNet
    Reference:
        ICCV2017: Image Super-Resolution Using Dense Skip Connections
        code: https://github.com/twtygqyy/pytorch-SRDenseNet/blob/master/srdensenet.py
    """
    def __init__(self,
                 nb_channel=3,
                 upscale_factor = 2,
                 base_kernel=128):
        super(SRDenseNet, self).__init__()
        kernels = [int(x * base_kernel) for x in [1, 2, 3, 4, 5, 6, 7, 8]]
        self.upscale_factor = upscale_factor
        self.relu = nn.PReLU()
        self.lowlevel = nn.Conv2d(nb_channel, kernels[0], kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv2d(kernels[0] + kernels[-1], kernels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction = nn.Conv2d(kernels[1], nb_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.denseblock1 = self.make_layer(_Dense_Block, kernels[0], kernels[0])
        self.denseblock2 = self.make_layer(_Dense_Block, kernels[1], kernels[0])
        self.denseblock3 = self.make_layer(_Dense_Block, kernels[2], kernels[0])
        self.denseblock4 = self.make_layer(_Dense_Block, kernels[3], kernels[0])
        self.denseblock5 = self.make_layer(_Dense_Block, kernels[4], kernels[0])
        self.denseblock6 = self.make_layer(_Dense_Block, kernels[5], kernels[0])
        self.denseblock7 = self.make_layer(_Dense_Block, kernels[6], kernels[0])
        self.denseblock8 = self.make_layer(_Dense_Block, kernels[7], kernels[0])
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(kernels[1], kernels[1], kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU(),)
        deconv_blocks = []
        for _ in range(self.upscale_factor - 1):
            deconv_blocks.append(nn.Sequential(
                    nn.ConvTranspose2d(kernels[1], kernels[1], kernel_size=2, stride=2, padding=0, bias=False),
                    nn.PReLU()))
        self.deconv = nn.Sequential(*deconv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, channel_in, base_kernel):
        layers = []
        layers.append(block(channel_in, base_kernel))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = self.relu(self.lowlevel(x))

        out = self.denseblock1(residual)
        concat = torch.cat([residual,out], 1)

        out = self.denseblock2(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock3(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock4(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock5(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock6(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock7(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock8(concat)
        out = torch.cat([concat,out], 1)

        out = self.bottleneck(out)

        out = self.deconv(out)

        out = self.reconstruction(out)

        return out

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error)
        return loss

if __name__ == "__main__":
    # Hyper Parameters
    img_col, img_row, nb_channel = 224, 224, 3
    upscale_factor = 2
    base_kernel = 32
    x = torch.FloatTensor(
        np.random.random((1, nb_channel, img_col, img_row)))
    model = SRDenseNet(nb_channel, upscale_factor, base_kernel)
    gen_y = model(x)
    print("VDSR->:")
    print(" Network input: ", x.shape)
    print("        output: ", gen_y.shape)
