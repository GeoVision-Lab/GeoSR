#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T18:36:20+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT
"""
import numpy as np
import torch
import torch.nn as nn


class FSRCNN(torch.nn.Module):
    """
    Superresolution using Super-Resolution Convolutional Neural Network
    Reference:
        ECCV2016: Accelerating the Super-Resolution Convolutional Neural Network
    """

    def __init__(self,
                 nb_channel=3,
                 upscale_factor=2,
                 d=64, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.relu = nn.PReLU()
        # Feature extraction
        self.part1 = nn.Conv2d(
            nb_channel, d, kernel_size=5, stride=1, padding=2)
        # Shrinking
        self.part2 = nn.Conv2d(d, s, kernel_size=1, stride=1, padding=0)
        # Non-linear Mapping
        self.layers = []
        for _ in range(m):
            self.layers.append(
                nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1))
        self.part3 = nn.Sequential(*self.layers)
        # Expanding
        self.part4 = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1, stride=1, padding=0))
        # Deconvolution
        self.part5 = nn.ConvTranspose2d(
            d, nb_channel, kernel_size=9, stride=upscale_factor, padding=4, output_padding=1)

    def forward(self, x):
        x = self.relu(self.part1(x))
        x = self.relu(self.part2(x))
        x = self.relu(self.part3(x))
        x = self.relu(self.part4(x))
        x = self.relu(self.part5(x))
        return x

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == "__main__":
    # Hyper Parameters
    img_col, img_row, nb_channel = 224, 224, 3
    upscale_factor = 2
    base_kernel = 64
    x = torch.FloatTensor(
        np.random.random((1, nb_channel, img_col, img_row)))

    model = FSRCNN(nb_channel, upscale_factor, base_kernel)
    gen_y = model(x)
    print("FSRCNN->:")
    print(" Network input: ", x.shape)
    print("        output: ", gen_y.shape)
