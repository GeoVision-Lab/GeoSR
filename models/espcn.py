#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

Superresolution using an efficient sub-pixel convolutional neural network    
Reference: 
    CVPR2016: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    code: https://github.com/pytorch/examples/tree/master/super_resolution
    PixelShuffle: https://blog.csdn.net/gdymind/article/details/82388068
"""
import sys
sys.path.append('./models')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, 
                 num_channel=3,
                 upscale_factor=2,
                 base_kernel=64):
        super(Net, self).__init__()
        kernels = [int(x * base_kernel) for x in [1, 1, 1/2]]

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(num_channel, kernels[0], kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(kernels[0], kernels[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(kernels[1], kernels[2], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(kernels[2], num_channel* upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
        
if __name__ == "__main__":
    # Hyper Parameters
    num_channel = 3
    upscale_factor = 2
    base_kernel = 64
    x = torch.FloatTensor(
            np.random.random((1, num_channel, 224, 224)))

    generator = Net(num_channel, upscale_factor, base_kernel)
    gen_y = generator(x)
    print("ESPCN->:")
    print(" Network input: ", x.shape)
    print("        output: ", gen_y.shape)


