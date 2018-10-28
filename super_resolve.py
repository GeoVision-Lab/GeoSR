from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from utils.runner import load_checkpoint

import numpy as np

import sys
import os
sys.path.append('/media/kaku/Work/geosr/')
DIR = os.path.dirname(os.path.abspath(__file__))
# Training settings
#parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
#parser.add_argument('--input_image', type=str, required=True, help='input image to use')
#parser.add_argument('--model', type=str, required=True, help='model file to use')
#parser.add_argument('--output_filename', type=str, help='where to save the output image')
#parser.add_argument('--cuda', action='store_true', help='use cuda')
#opt = parser.parse_args()

#print(opt)

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--band_mode', type=str, default='Y', choices=['Y', 'YCbCr', 'RGB'], help="band mode")
parser.add_argument('--data_dir', type=str, default=os.path.join(DIR, 'dataset','map-rand','images','test'), help="data directory")
parser.add_argument('--crop_size', type=int, default=224, help='crop size from each data. Default=224 (same to image size)')
parser.add_argument('--nb_channel', type=int, default=1, help="input image band")
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--aug', type=lambda x: (str(x).lower() == 'true'), default=True, help='data augmentation or not') 
parser.add_argument('--aug_mode', type=str, default='c', choices=['a', 'b', 'c', 'd', 'e'], 
                    help='data augmentation mode: a, b, c, d, e')
parser.add_argument('--base_kernel', type=int, default=64, help="base kernel")
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--testbatch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=6, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')

parser.add_argument('--trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                    help='trigger type for logging')
parser.add_argument('--interval', type=int, default=2,
                    help='interval for logging')
parser.add_argument('--cuda', type=lambda x: (str(x).lower() == 'true'), default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--model_name', type=str, default='up4_ESPCN_epoch_8_Oct28_11.pth', help='model name')
parser.add_argument('--result_dir', type=str, default=os.path.join(DIR, 'result'), help='result dir')
args = parser.parse_args()
#result = main(args)

model_name = args.model_name
model =  load_checkpoint(model_name)  
img_files = os.listdir(args.data_dir)
print('===> Testing')
for img_file in img_files:    
    img_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(args.data_dir, img_file)
    _model_name = os.path.splitext(model_name)[0]
    output_file = img_name + '_' + _model_name + os.path.splitext(img_file)[1]
    output_path = os.path.join(args.result_dir, output_file)
    img = Image.open(img_path).convert('YCbCr')
    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    if args.cuda:
        model = model.cuda()
        input = input.cuda()
    
    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')   
    
    out_img.save(output_path)
    