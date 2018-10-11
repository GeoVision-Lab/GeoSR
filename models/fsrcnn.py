#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T18:36:20+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

Superresolution using Super-Resolution Convolutional Neural Network  
Reference: 
    ECCV2016: Accelerating the Super-Resolution Convolutional Neural Network
"""
import sys
sys.path.append('./models')
import numpy as np
import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, 
                 num_channels=3,
                 upscale_factor=2,
                 d=64, s=12, m=4):
        super(Net, self).__init__()
        self.relu = nn.PReLU()
        # Feature extraction
        self.part1 = nn.Conv2d(num_channels, d, kernel_size=5, stride=1, padding=2)
        # Shrinking
        self.part2 = nn.Conv2d(d, s, kernel_size=1, stride=1, padding=0)
        # Non-linear Mapping
        self.layers = []
        for _ in range(m):
            self.layers.append(nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1))
        self.part3 = nn.Sequential(*self.layers)
        # Expanding
        self.part4 = nn.Sequential(nn.Conv2d(s, d, kernel_size=1, stride=1, padding=0))
        # Deconvolution
        self.part5 = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=upscale_factor, padding=4, output_padding=1)

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
    num_channel = 3
    upscale_factor = 2
    base_kernel = 64
    x = torch.FloatTensor(
            np.random.random((1, num_channel, 224, 224)))

    generator = Net(num_channel, upscale_factor, base_kernel)
    gen_y = generator(x)
    print("FSRCNN->:")
    print(" Network input: ", x.shape)
    print("        output: ", gen_y.shape)