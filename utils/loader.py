#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

Load training, test, and validation data from dataset/data_dir
"""
import os
import sys
sys.path.append('./utils')
import torch.utils.data as data
import argparse
from PIL import Image
from preprocessor import input_transform, target_transform

Utils_DIR = os.path.dirname(os.path.abspath(__file__))

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])

def load_img(filepath):
    """
    the original code only choose one band
    """
#    img = imread(filepath)
    img = Image.open(filepath).convert('YCbCr')
#    y, _, _ = img.split()
    y = img
    return y

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_training_set(data_dir, crop_size, upscale_factor):
    root_dir = os.path.join(data_dir,'image')
    train_dir = os.path.join(root_dir, "train")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

def get_val_set(data_dir, crop_size, upscale_factor):
    root_dir = os.path.join(data_dir,'image')
    val_dir = os.path.join(root_dir, "val")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(val_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

def get_test_set(data_dir, crop_size, upscale_factor):
    root_dir = os.path.join(data_dir,'image')
    test_dir = os.path.join(root_dir, "test")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--data_dir', type=str, default=os.path.join(Utils_DIR, '../dataset','map-rand'), help="data directory")
    parser.add_argument('--crop_size', type=int, default=224, help='crop size from each data. Default=224 (same to image size)')
    parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
    opt = parser.parse_args()
    print(opt)
    
    train_set = get_training_set(opt.data_dir, opt.crop_size, opt.upscale_factor)
    print('Training number: {}'.format(len(train_set.image_filenames)))
    print('Input transform:')
    print(train_set.input_transform)
    print('Target transform:')
    print(train_set.target_transform)


