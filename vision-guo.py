# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T14:47:02+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

Vision Tools
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import torch

Utils_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(Utils_DIR, '../models'))

class show_compare_sigle():
    """
    Illustrate and compare sigle hr, lr, and sr images
    """
    def __init__(self, images, titles):
        """
        Parameters
        ----------
            images: list
                eg: [hr, lr, sr]
            titles: lsit
                eg: ['high resolution', 'low resolution', 'super resolution']
        """
        self.images = images
        self.titles = titles
    def lr_sr(self):
        images = self.images
        titles = self.titles
        plt.figure(figsize=(6, 4))
        for i in range(len(images)):
            plt.subplot(1, len(images), i+1)
            plt.title(titles[i])
#            plt.xticks([])
#            plt.yticks([])
            plt.imshow(images[i])
        plt.tight_layout()
        plt.show()
    def hr_lr_sr(self):
        images = self.images
        titles = self.titles
        plt.figure(figsize=(6, 4))
        for i in range(len(images)):
            plt.subplot(1, len(images), i+1)
            plt.title(titles[i])
            plt.imshow(images[i])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from fsrcnn import FSRCNN

    img_path = './data/image.png'
    upscale_factor = 2

    img = imread(img_path)
    nb_channel = img.shape[-1]
    img_torch = torch.FloatTensor(np.expand_dims(np.transpose(img,(-1,0,1)),0))

    generator = FSRCNN(nb_channel, upscale_factor)
    gen_img = generator(img_torch)
    gen_img = np.squeeze(gen_img.detach().numpy())
    sr_img = np.transpose(gen_img,(1,2,0))

    images = [img, sr_img]
    titles = ['low resolution', 'super resolution']
    show_compare = show_compare_sigle(images, titles)
    show_compare.lr_sr()
