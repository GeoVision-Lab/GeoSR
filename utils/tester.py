#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT
"""

import os
import sys
sys.path.append('./utils')
sys.path.append('..')
import time
import torch
from utils import metrics
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from loader import get_real_test_set

Utils_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(Utils_DIR, '..')
Logs_DIR = os.path.join(Utils_DIR, '../logs')
Checkpoint_DIR = os.path.join(Utils_DIR, '../model_zoo')
Result_DIR = os.path.join(Utils_DIR, '../result')

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def load_checkpoint(name):
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, name)
                          ), "{} not exists.".format(name)
    print("Loading model: {}".format(name))
    return torch.load("{}/{}".format(Checkpoint_DIR, name))

def load_middle_checkpoint(middle_checkpoint_dir, name):
    assert os.path.exists("{}/{}/{}".format(Checkpoint_DIR, middle_checkpoint_dir, name)
                          ), "{} not exists.".format(name)
    print("Loading middel checkpoint: {}".format(name))
    return torch.load("{}/{}/{}".format(Checkpoint_DIR, middle_checkpoint_dir, name))

class Result_Generator(object):
    def __init__(self, args, img_path, model):
        self.args = args
        self.img_path = img_path
        self.model = model
#        self.output_path = output_path
    def Y_to_RGB(self):
        img_path = self.img_path
        model = self.model
        img = Image.open(img_path).convert('YCbCr')
        y, cb, cr = img.split()
        img_to_tensor = ToTensor()
        input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
        if self.args.cuda:
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
        return out_img
    
    def YCbCr_to_RGB(self):
        img_path = self.img_path
        model = self.model
        img = Image.open(img_path).convert('YCbCr')
        img_to_tensor = ToTensor()
        input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0])
        if self.args.cuda:
            model = model.cuda()
            input = input.cuda()
        out = model(input)
        out = torch.squeeze(out)
        out_img = out.cpu()
        out_img = out_img.detach().numpy()
        out_img *= 255.0
        out_img = out_img.clip(0, 255)
        out_img = np.transpose(out_img, (1,2,0)) 
        out_img = Image.fromarray(np.uint8(out_img), mode='YCbCr')
        out_img = out_img.convert('RGB')
        return out_img     
    
    def RGB_to_RGB(self):
        img_path = self.img_path
        model = self.model
        img = Image.open(img_path).convert('RGB')
        img_to_tensor = ToTensor()
        input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0])
        
        if self.args.cuda:
            model = model.cuda()
            input = input.cuda()
        out = model(input)
        out = torch.squeeze(out)        
        out = out.cpu()
        out_img = out.detach().numpy
        return out_img

    def __call__(self):
        if self.args.band_mode == 'Y':
            return self.Y_to_RGB()
        if self.args.band_mode == 'YCbCr':
            return self.YCbCr_to_RGB()
        if self.args.band_mode == 'RGB':
            return self.RGB_to_RGB()
        
class Base(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.logs = []
        self.test_dir = args.test_dir
        """
        metrics
        psnr, nrmse, ssim, vifp, fsim
        """
        self.headers = ["ip", "model", "psnr", "ssim", "nrmse"]

    def logging(self, result_log, verbose=False):
        self.logs.append([self.ip, os.path.splitext(self.args.model_name)[0]] +
                         result_log)
        if verbose:
            print("psnr:{:0.3f}, ssim:{:0.3f}, nrmse:{:0.3f}"
                  .format(result_log[0], result_log[1], result_log[2]))

    def save_result_log(self):
        self.result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(self.args.test_dir.split('/')[-3:]), 'diff_model')
        
        if not os.path.exists(self.result_save_dir):
            os.makedir(self.result_save_dir)
        self.logs = pd.DataFrame(self.logs,
                                 columns=self.headers)
        self.logs.to_csv(os.path.join(self.result_save_dir, "result_log.csv"), index=False)

    def save_result_log_new(self):
        self.result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(self.args.test_dir.split('/')[-3:]), 'diff_model')
        
        if not os.path.exists(self.result_save_dir):
            os.makedir(self.result_save_dir)
        self.logs = pd.DataFrame(self.logs,
                                 columns=self.headers)

        if os.path.exists(os.path.join(self.result_save_dir, 'result_log.csv')):
            print(1111111111111111)
            result_logs = pd.read_csv(os.path.join(self.result_save_dir, 'result_log.csv'))
            print(len(result_logs))
            result_logs.append(self.logs)
            print(len(result_logs))
            print(len(self.logs))
        else:
            print(2222222222222222)
            result_logs = self.logs
        result_logs.to_csv(os.path.join(self.result_save_dir, "result_log.csv"), index=False)


#        if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}.csv".format(split))):
#            logs = pd.read_csv(os.path.join(
#                Logs_DIR, 'statistic', "{}.csv".format(split)))
#        else:
#            logs = pd.DataFrame([])
#        logs = logs.append(cur_log, ignore_index=True)
#        logs.to_csv(os.path.join(Logs_DIR, 'statistic',
#                                 "{}.csv".format(split)), index=False, float_format='%.3f')


        
class Tester(Base):
    def testing_model(self, model_name, test_dir):
        args = self.args
#        model_name = args.model_name       
        model = load_checkpoint(model_name)
        
        print('===> Testing')
        if args.ground_truth:
            test_set = get_real_test_set(args.band_mode, args.test_dir, args.aug, args.aug_mode, args.crop_size, args.upscale_factor)
            self.evaluating(model, model_name, test_set)
        else:
            img_files = os.listdir(test_dir)    
            result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(args.test_dir.split('/')[-3:]), 'diff_model')
            if not os.path.exists(result_save_dir):
                os.makedirs(result_save_dir)
            
            for img_file in img_files:    
                img_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(args.test_dir, img_file)
                _model_name = os.path.splitext(model_name)[0]
                output_file = img_name + '_' + _model_name + os.path.splitext(img_file)[1]
                output_path = os.path.join(result_save_dir, output_file)
                result_generator = Result_Generator(args, img_path, model)
                out_img = result_generator()
                out_img.save(output_path)
                
    def testing_middle_checkpoint(self, model_name, test_dir):
        args = self.args
        checkpoint_dir = os.path.splitext(model_name)[0]
        checkpoint_path = os.path.join(DIR, 'model_zoo', checkpoint_dir)
        print('===> Testing')
        for checkpoint_name in os.listdir(checkpoint_path):
            model =  load_middle_checkpoint(checkpoint_dir, checkpoint_name)  
            img_files = os.listdir(args.test_dir)
            result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(args.test_dir.split('/')[-3:]), 'middle_checkpoint',checkpoint_dir)
            if not os.path.exists(result_save_dir):
                os.makedirs(result_save_dir)
            for img_file in img_files:    
                img_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(args.test_dir, img_file)
                _checkpoint_name = os.path.splitext(checkpoint_name)[0]
                output_file = img_name + '_' + _checkpoint_name + os.path.splitext(img_file)[1]
                output_path = os.path.join(result_save_dir, output_file)
                result_generator = Result_Generator(args, img_path, model)
                out_img = result_generator()
                out_img.save(output_path)
                
    def evaluating(self, model, model_name, dataset):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
#        oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
        """
        metrics
        """
#        psnr, nrmse, ssim, vifp, fsim
        psnr_all, nrmse_all, ssim_all = 0, 0, 0
        model.eval()
        
        img_files = sorted(os.listdir(args.test_dir))
        result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(args.test_dir.split('/')[-3:]), 'diff_model')
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)
        
        for img_file in img_files:    
            img_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(args.test_dir, img_file)
            _model_name = os.path.splitext(model_name)[0]
            output_file = img_name + '_' + _model_name + os.path.splitext(img_file)[1]
            output_path = os.path.join(result_save_dir, output_file)
            result_generator = Result_Generator(args, img_path, model)
            out_img = result_generator()
            out_img.save(output_path)        
              
        data_loader = DataLoader(dataset, 1, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.testbatch_size

        for step in range(steps):
            x, y = next(batch_iterator)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
            # calculate pixel accuracy of generator
            """
            metrics
            """
            gen_y = model(x)
            psnr = metrics.psnr(gen_y, y)
            psnr_all += psnr
            nrmse = metrics.nrmse(gen_y, y)
            nrmse_all += psnr
            ssim = metrics.ssim(gen_y, y)
            ssim_all += ssim
            self.ip = img_files[step]
            """
            metrics
            """
            self.result_log = [round(idx, 3) for idx in [psnr, nrmse, ssim]]
            self.logging(self.result_log)
        self.save_result_log_new()
