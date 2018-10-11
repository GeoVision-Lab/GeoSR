#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T16:23:01+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

Superresolution using a Deep Convolutional Network    
Reference: 
    ECCV2014: Learning a Deep Convolutional Network for Image Super-Resolution
"""
import sys
sys.path.append('./models')
import numpy as np
import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, 
                 num_channel=3,
                 upscale_factor=2,
                 base_kernel=64):
        super(Net, self).__init__()
        kernels = [int(x * base_kernel) for x in [1, 1/2]]
        
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(num_channel, kernels[0], kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(kernels[0], kernels[1], kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(kernels[1], num_channel, kernel_size=5, stride=1, padding=2)        

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
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
    print("SRCNN->:")
    print(" Network input: ", x.shape)
    print("        output: ", gen_y.shape)