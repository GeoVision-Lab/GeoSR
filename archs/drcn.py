#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Shiba-lab
@License: MIT
"""
import numpy as np
import torch
import torch.nn as nn
#from torch.autograd import Variable

class DRCN(torch.nn.Module):
    """
    Superresolution using Deeply-Recursive Convolutional Network
    Different from other code(mistake maybe?) 
        https://github.com/icpm/super-resolution/blob/master/DRCN/model.py
    Reference:
        CVPR2016: Deeply-Recursive Convolutional Network for Image Super-Resolution
    """
    def __init__(self,
                 nb_channel=3, 
                 base_kernel=64,
                 num_recursions=16):
        super(DRCN, self).__init__()
        self.num_recursions = num_recursions
        # embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(nb_channel, base_kernel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_kernel, base_kernel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # conv block of inference layer
        self.conv_block = nn.Sequential(nn.Conv2d(base_kernel, base_kernel, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))
        # reconstruction layer
        self.reconstruction_layer = nn.Sequential(
            nn.Conv2d(base_kernel, base_kernel, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(base_kernel, nb_channel, kernel_size=3, stride=1, padding=1)
        )

        # initial w
        self.w_init = torch.ones(num_recursions) / num_recursions
#        self.w = Variable(self.w_init.cuda(), requires_grad=True)
        self.w = self.w_init
    def forward(self, x):
        # embedding layer
        h0 = self.embedding_layer(x)
        # recursions
        h = [h0]
        for d in range(self.num_recursions):
            h.append(self.conv_block(h[d]))
        y_d_ = []
        out = []
        out_sum = 0
        for d in range(self.num_recursions):
            y_d_.append(self.reconstruction_layer(h[d+1]))
            # skip connection
            out.append(torch.add(y_d_[d], x))
        out_sum += torch.mul(out[d], self.w[d])
        #normalization
        final_out = torch.mul(out_sum, 1.0 / (torch.sum(self.w)))
        return out, final_out
#        for d in range(num_recursions):
#            y_d_.append(self.reconstruction_layer(h[d+1]))
#            out_sum += torch.mul(y_d_[d], self.w[d])
#        out_sum = torch.mul(out_sum, 1.0 / (torch.sum(self.w)))
#
#        # skip connection
#        final_out = torch.add(out_sum, x)
#        return y_d_, final_out

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
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
    num_recursions = 16
    x = torch.FloatTensor(
        np.random.random((1, nb_channel, img_col, img_row)))

    model = DRCN(nb_channel, base_kernel, num_recursions)
    gen_y = model(x)
    print("DRCN->:")
    print(" Network input: ", x.shape)
    print("        output: ", gen_y[-1].shape)