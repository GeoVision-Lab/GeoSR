#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

Load training, test, and validation data from dataset/data_dir, \
    with data augmentation method in DataAug
"""
import os
import sys
sys.path.append('./utils')
import torch.utils.data as data
import argparse
from PIL import Image
from utils.preprocessor import DataAug
import torchvision.transforms.functional as TF

Utils_DIR = os.path.dirname(os.path.abspath(__file__))

def load_img(filepath):
    """
    the original code only choose one band
    """
#    img = imread(filepath)
    img = Image.open(filepath).convert('YCbCr')
#    y, _, _ = img.split()
    y = img
    return y

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=True, aug_mode='a', crop_size=224, upscale_factor=2):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        self.transform = transform
        self.trans_mode = aug_mode
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        """
        data.Dataset can make index range from 0 to len(image_filenames)
        """
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.transform:
#            input, target = DataAug(input, target, self.crop_size, self.upscale_factor)
            data_aug = DataAug(input, target, self.trans_mode, self.crop_size, self.upscale_factor)
            input, target = data_aug()
        else:
            input = TF.center_crop(input, (self.crop_size, self.crop_size))
            input = TF.resize(input, (self.crop_size // self.upscale_factor, self.crop_size // self.upscale_factor))
            input = TF.to_tensor(input)

            target = TF.center_crop(target, (self.crop_size, self.crop_size))
            target = TF.to_tensor(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_training_set(data_dir, aug, aug_mode, crop_size, upscale_factor):
    root_dir = os.path.join(data_dir,'images')
    train_dir = os.path.join(root_dir, "train")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             transform=aug,
                             aug_mode=aug_mode,
                             crop_size=crop_size, 
                             upscale_factor=upscale_factor)

def get_val_set(data_dir, aug, aug_mode, crop_size, upscale_factor):
    root_dir = os.path.join(data_dir,'images')
    val_dir = os.path.join(root_dir, "val")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(val_dir,
                             transform=aug,
                             aug_mode=aug_mode,
                             crop_size=crop_size, 
                             upscale_factor=upscale_factor)

def get_test_set(data_dir, aug, aug_mode, crop_size, upscale_factor):
    root_dir = os.path.join(data_dir,'images')
    test_dir = os.path.join(root_dir, "test")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             transform=aug,
                             aug_mode=aug_mode,
                             crop_size=crop_size, 
                             upscale_factor=upscale_factor)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--data_dir', type=str, default=os.path.join(Utils_DIR, '../dataset','map-rand'), help="data directory")
    parser.add_argument('--crop_size', type=int, default=224, help='crop size from each data. Default=224 (same to image size)')
    parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
    parser.add_argument('--aug', type=lambda x: (str(x).lower() == 'true'), default=True, help='data augmentation or not') 
    parser.add_argument('--aug_mode', type=str, default='c', choices=['a', 'b', 'c', 'd', 'e'], 
                        help='data augmentation mode: a, b, c, d, e')
    opt = parser.parse_args()
    print(opt)
    
    train_set = get_training_set(opt.data_dir, opt.aug, opt.aug_mode,opt.crop_size, opt.upscale_factor)
    print('Training number: {}'.format(len(train_set.image_filenames)))
#    print('Input transform:')
#    print(train_set.input_transform)
#    print('Target transform:')
#    print(train_set.target_transform)


