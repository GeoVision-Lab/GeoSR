#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

preprocessor.py
    Data augmentation
"""
import os
import sys
sys.path.append('./utils')
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, FiveCrop

Utils_DIR = os.path.dirname(os.path.abspath(__file__))

def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
#        FiveCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),])

def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),])

