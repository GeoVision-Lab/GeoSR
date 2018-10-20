#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

preprocessor.py
    Scalable data augmentation methods 
"""
import os
import sys
sys.path.append('./utils')
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, FiveCrop, RandomCrop
import torchvision.transforms.functional as TF
import torch.utils.data as data
import random
from PIL import Image

Utils_DIR = os.path.dirname(os.path.abspath(__file__))

class DataAug(object):
    """
    Scalable data augmentation methods
    Add self augmentation method at the end
    """
    def __init__(self, image, mask, trans_mode, crop_size, upscale_factor):
        
        self.image = image
        self.mask = mask
        self.trans_mode = trans_mode
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

    def transform_a(self, image, mask):
        """
        Center crop
        """
        # Center crop
        image = TF.center_crop(image, (self.crop_size,self.crop_size))
        mask = TF.center_crop(image, (self.crop_size,self.crop_size))
        
        # resize
        image = image
        image = TF.resize(image, (self.crop_size // self.upscale_factor,self.crop_size // self.upscale_factor))
        
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask
    
    def transform_b(self, image, mask):
        """
        Center crop + Random horizontal flipping
        """        
        # Center crop
        image = TF.center_crop(image, (self.crop_size,self.crop_size))
        mask = TF.center_crop(image, (self.crop_size,self.crop_size))
        
        # resize
        image = image
        image = TF.resize(image, (self.crop_size // self.upscale_factor,self.crop_size // self.upscale_factor))
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask
    
    def transform_c(self, image, mask):
        """
        Center crop + Random horizontal flipping + Random vertical flipping
        """        
        # Center crop
        image = TF.center_crop(image, (self.crop_size,self.crop_size))
        mask = TF.center_crop(image, (self.crop_size,self.crop_size))
        
        # resize
        image = image
        image = TF.resize(image, (self.crop_size // self.upscale_factor,self.crop_size // self.upscale_factor))
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def transform_d(self, image, mask):
        """
        Random crop + Random horizontal flipping + Random vertical flipping
        """  
        # Random crop
        i, j, h, w = RandomCrop.get_params(
            image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        # resize
        image = image
        image = TF.resize(image, (self.crop_size // self.upscale_factor,self.crop_size // self.upscale_factor))
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def transform_e(self, image, mask):
        """
        Deside by your self
        """  
        raise TypeError('Please decide your own augmentation method')
    
    def __call__(self):
        if self.trans_mode == 'a':
            image, mask = self.transform_a(self.image, self.mask)
        if self.trans_mode == 'b':
            image, mask = self.transform_b(self.image, self.mask)
        if self.trans_mode == 'c':
            image, mask = self.transform_c(self.image, self.mask)
        if self.trans_mode == 'd':
            image, mask = self.transform_d(self.image, self.mask)
        if self.trans_mode == 'e':
            image, mask = self.transform_e(self.image, self.mask)
        return image, mask

