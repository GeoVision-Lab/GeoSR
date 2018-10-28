#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T16:23:01+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT
"""
import numpy as np
import torch
import torch.nn as nn


class SRCNN(torch.nn.Module):
    """
    Superresolution using a Deep Convolutional Network
    Reference:
        ECCV2014: Learning a Deep Convolutional Network for Image Super-Resolution
    """

    def __init__(self,
                 nb_channel=3,
                 base_kernel=64):
        super(SRCNN, self).__init__()
        kernels = [int(x * base_kernel) for x in [1, 1 / 2]]

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(
            nb_channel, kernels[0], kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(
            kernels[0], kernels[1], kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(
            kernels[1], nb_channel, kernel_size=5, stride=1, padding=2)

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
    img_col, img_row, nb_channel = 224, 224, 3
    base_kernel = 64
    x = torch.FloatTensor(
        np.random.random((1, nb_channel, img_col, img_row)))

    model = SRCNN(nb_channel, base_kernel)
    gen_y = model(x)
    print("SRCNN->:")
    print(" Network input: ", x.shape)
    print("        output: ", gen_y.shape)
