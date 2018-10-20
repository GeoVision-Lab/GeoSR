#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT

Geosr - A Computer Vision Package for Remote Sensing Image Super Resolution
"""
from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.espcn import ESPCN
from utils.loader import get_training_set, get_test_set

DIR = os.path.dirname(os.path.abspath(__file__))

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--data_dir', type=str, default=os.path.join(DIR, 'dataset','map-rand'), help="data directory")
parser.add_argument('--crop_size', type=int, default=224, help='crop size from each data. Default=224 (same to image size)')
parser.add_argument('--nb_channel', type=int, default=3, help="input image band")
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--aug', type=lambda x: (str(x).lower() == 'true'), default=True, help='data augmentation or not') 
parser.add_argument('--aug_mode', type=str, default='c', choices=['a', 'b', 'c', 'd', 'e'], 
                    help='data augmentation mode: a, b, c, d, e')
parser.add_argument('--base_kernel', type=int, default=64, help="base kernel")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
    
torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.aug, opt.aug_mode, opt.crop_size, opt.upscale_factor)
#train_set = get_training_set(opt.data_dir, opt.upscale_factor)
test_set = get_test_set(opt.data_dir, opt.aug, opt.aug_mode, opt.crop_size, opt.upscale_factor)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
    
print('===> Building model')
model = ESPCN(nb_channel=opt.nb_channel, upscale_factor=opt.upscale_factor, base_kernel=opt.base_kernel).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            
            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)