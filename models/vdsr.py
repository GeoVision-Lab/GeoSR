#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T18:36:20+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Shiba-lab
@License: MIT
"""
import numpy as np
import torch
import torch.nn as nn

class VDSR(nn.Module):
    """
    Superresolution using Very Deep Convolutional Networks
    The network takes an interpolated low-resolution image as input and predicts image details.
    Reference:
        CVPR2016: Accurate Image Super-Resolution Using Very Deep Convolutional Networks
    """
    def __init__(self, 
                 nb_channels=3,
                 base_kernel=64, 
                 num_residuals=18):
        
        super(VDSR, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.input_conv = nn.Sequential(nn.Conv2d(nb_channels, base_kernel, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.ReLU(inplace=True))

        conv_blocks = []
        for _ in range(num_residuals):
            conv_blocks.append(nn.Sequential(nn.Conv2d(base_kernel, base_kernel, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.ReLU(inplace=True)))
        self.residual_layers = nn.Sequential(*conv_blocks)

        self.output_conv = nn.Conv2d(base_kernel, nb_channels, kernel_size=3, stride=1, padding=1, bias=False)
 
    def forward(self, x):
        residual = x
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        x = torch.add(x, residual)
        return x

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
            
if __name__ == "__main__":
    # Hyper Parameters
    img_col, img_row, nb_channel = 224, 224, 3
    base_kernel = 64
    x = torch.FloatTensor(
        np.random.random((1, nb_channel, img_col, img_row)))

    model = VDSR(nb_channel, base_kernel)
    gen_y = model(x)
    print("SRCNN->:")
    print(" Network input: ", x.shape)
    print("        output: ", gen_y.shape)