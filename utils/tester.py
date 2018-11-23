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
import torch
from utils import metrics
import pandas as pd
import warnings
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF


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

class Result_Generator_Without_Truth(object):
    def __init__(self, args, img_path, model):
        self.args = args
        self.img_path = img_path
        self.model = model
#        self.output_path = output_path
    def Y_to_RGB(self):
        img_path = self.img_path
        model = self.model
        img = Image.open(img_path).convert('YCbCr')
        if self.args.interpolation:
            img = img.resize((img.size[0]*self.args.upscale_factor,img.size[1]*self.args.upscale_factor), resample = Image.BICUBIC)
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

class Result_Generator_With_Truth(object):
    def __init__(self, args, img_path, model):
        self.args = args
        self.img_path = img_path
        self.model = model
#        self.output_path = output_path
    def Y_to_RGB(self):
        args = self.args
        img_path = self.img_path
        model = self.model
        
        img_to_tensor = ToTensor()
        hr_img = Image.open(img_path).convert('YCbCr')
        hr_img_y = hr_img.split()[0]
        hr_img_y_tensor = img_to_tensor(hr_img_y).view(1, -1, hr_img_y.size[1], hr_img_y.size[0])
        hr_img_Cb = hr_img.split()[1]
        hr_img_Cr = hr_img.split()[2]

        if args.interpolation:
            args.upscale_factor = 1
        
        lr_img_y = TF.resize(hr_img_y, ( hr_img_y.size[0]// args.upscale_factor, hr_img_y.size[1] // args.upscale_factor))
        lr_img_y_tensor = img_to_tensor(lr_img_y).view(1, -1, lr_img_y.size[1], lr_img_y.size[0])
        input = lr_img_y_tensor
        
        if args.cuda:
            model = model.cuda()
            input = input.cuda()                    
        sr_img_y_tensor = model(input)
        sr_img_y_tensor = sr_img_y_tensor.cpu()        
        
        """
        metrics
        """ 
        psnr = metrics.psnr(sr_img_y_tensor, hr_img_y_tensor)
        nrmse = metrics.nrmse(sr_img_y_tensor, hr_img_y_tensor)
        ssim = metrics.ssim(sr_img_y_tensor, hr_img_y_tensor)
        
        sr_img_y = sr_img_y_tensor[0].detach().numpy()        
        sr_img_y *= 255.0
        sr_img_y = sr_img_y.clip(0, 255)
        sr_img_y = Image.fromarray(np.uint8(sr_img_y[0]), mode='L')        
        sr_img_Cb = hr_img_Cb
        sr_img_Cr = hr_img_Cr
        sr_img = Image.merge('YCbCr', [sr_img_y, sr_img_Cb, sr_img_Cr]).convert('RGB')            

        return sr_img, (psnr, ssim, nrmse)

    def __call__(self):
        if self.args.band_mode == 'Y':
            return self.Y_to_RGB()
        if self.args.band_mode == 'YCbCr':
            return self.YCbCr_to_RGB()
        if self.args.band_mode == 'RGB':
            return self.RGB_to_RGB()

def lerp_img_evaluation(hr_img, lerp_img):
    lerp_img_y = lerp_img.split()[0]
    hr_img_y =hr_img.split()[0]
    img_to_tensor = ToTensor()
    lerp_img_y_tensor = img_to_tensor(lerp_img_y).view(1, -1, lerp_img_y.size[1], lerp_img_y.size[0])
    hr_img_y_tensor = img_to_tensor(hr_img_y).view(1, -1, hr_img_y.size[1], hr_img_y.size[0])
    
    psnr = metrics.psnr(lerp_img_y_tensor, hr_img_y_tensor)
    nrmse = metrics.nrmse(lerp_img_y_tensor, hr_img_y_tensor)
    ssim = metrics.ssim(lerp_img_y_tensor, hr_img_y_tensor)
    return psnr, nrmse, ssim
        
        
class Base(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.logs = []
        self.lerp_logs = []
        self.middle_logs = []
        self.avg_logs = []
        self.lerp_avg_logs = []
        self.middle_avg_logs = []
        self.test_dir = args.test_dir
        """
        metrics
        psnr, nrmse, ssim, vifp, fsim
        """
        self.headers = ["ip", "model", "psnr", "ssim", "nrmse"]
        self.avg_headers = ["model", "psnr_avg", "ssim_avg", "nrmse_avg"]

    def logging(self, result_log, verbose=False):
        self.logs.append([self.ip, os.path.splitext(self.args.test_model_name)[0]] +
                         result_log)
        if verbose:
            print("psnr:{:0.3f}, ssim:{:0.3f}, nrmse:{:0.3f}"
                  .format(result_log[0], result_log[1], result_log[2]))

    def save_result_log(self):
        self.result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(self.args.test_dir.split('/')[-3:]), 'diff_model', 'with_truth')

        if not os.path.exists(self.result_save_dir):
            os.makedir(self.result_save_dir)
        cur_log = pd.DataFrame(self.logs,
                                 columns=self.headers)

        if os.path.exists(os.path.join(self.result_save_dir, 'result_log.csv')):
            result_logs = pd.read_csv(os.path.join(self.result_save_dir, 'result_log.csv'))
            result_logs = result_logs.append(cur_log, ignore_index=True)
        else:
            result_logs = cur_log
        result_logs = result_logs.sort(['ip'])
        result_logs.to_csv(os.path.join(self.result_save_dir, "result_log.csv"), index=False)

    def logging_avg(self, result_avg_log, verbose=True):
        self.avg_logs.append([os.path.splitext(self.args.test_model_name)[0]] +
                         result_avg_log)
        if verbose:
            print("psnr_avg:{:0.3f}, ssim_avg:{:0.3f}, nrmse_avg:{:0.3f}"
                  .format(result_avg_log[0], result_avg_log[1], result_avg_log[2]))

    def save_result_avg_log(self):
        self.result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(self.args.test_dir.split('/')[-3:]), 'diff_model', 'with_truth')
        
        if not os.path.exists(self.result_save_dir):
            os.makedir(self.result_save_dir)
        cur_log = pd.DataFrame(self.avg_logs,
                                 columns=self.avg_headers)

        if os.path.exists(os.path.join(self.result_save_dir, 'result_avg_log.csv')):
            result_logs = pd.read_csv(os.path.join(self.result_save_dir, 'result_avg_log.csv'))
            result_logs = result_logs.append(cur_log, ignore_index=True)
        else:
            result_logs = cur_log
        result_logs = result_logs.sort(['model'])
        result_logs.to_csv(os.path.join(self.result_save_dir, "result_avg_log.csv"), index=False)
    
    """
    middle checkpoint
    """
    
    def middle_logging(self, result_log, checkpoint_name, verbose=False):
        self.middle_logs.append([self.ip, checkpoint_name] +
                         result_log)
        if verbose:
            print("psnr:{:0.3f}, ssim:{:0.3f}, nrmse:{:0.3f}"
                  .format(result_log[0], result_log[1], result_log[2]))
            
    def logging_middle_avg(self, result_avg_log, checkpoint_name, verbose=True):
        self.middle_avg_logs.append([checkpoint_name] + result_avg_log)
        if verbose:
            print("psnr_avg:{:0.3f}, ssim_avg:{:0.3f}, nrmse_avg:{:0.3f}"
                  .format(result_avg_log[0], result_avg_log[1], result_avg_log[2]))        
            
    def save_middle_result_log(self):
        self.result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(self.args.test_dir.split('/')[-3:]), 'middle_checkpoint',self.checkpoint_dir,'with_truth')

        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        cur_log = pd.DataFrame(self.middle_logs,
                                 columns=self.headers)

        if os.path.exists(os.path.join(self.result_save_dir, 'result_log.csv')):
            result_logs = pd.read_csv(os.path.join(self.result_save_dir, 'result_log.csv'))
            result_logs = result_logs.append(cur_log, ignore_index=True)
        else:
            result_logs = cur_log
        result_logs = result_logs.sort(['ip'])
        result_logs.to_csv(os.path.join(self.result_save_dir, "result_log.csv"), index=False)

    def save_middle_result_avg_log(self):
        self.result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(self.args.test_dir.split('/')[-3:]), 'middle_checkpoint',self.checkpoint_dir,'with_truth')
        
        if not os.path.exists(self.result_save_dir):
            os.makedir(self.result_save_dir)
        cur_log = pd.DataFrame(self.middle_avg_logs,
                                 columns=self.avg_headers)

        if os.path.exists(os.path.join(self.result_save_dir, 'result_avg_log.csv')):
            result_logs = pd.read_csv(os.path.join(self.result_save_dir, 'result_avg_log.csv'))
            result_logs = result_logs.append(cur_log, ignore_index=True)
        else:
            result_logs = cur_log
        result_logs = result_logs.sort(['model'])
        result_logs.to_csv(os.path.join(self.result_save_dir, "result_avg_log.csv"), index=False)

    """
    linear interpolation
    """
    def lerp_logging(self, result_log, lerp_name, verbose=False):
        self.lerp_logs.append([self.ip, lerp_name] +
                         result_log)
        if verbose:
            print("psnr:{:0.3f}, ssim:{:0.3f}, nrmse:{:0.3f}"
                  .format(result_log[0], result_log[1], result_log[2]))

    def save_lerp_result_log(self):
        self.result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(self.args.test_dir.split('/')[-3:]), 'diff_model', 'with_truth')

        if not os.path.exists(self.result_save_dir):
            os.makedir(self.result_save_dir)
        cur_log = pd.DataFrame(self.lerp_logs,
                                 columns=self.headers)

        if os.path.exists(os.path.join(self.result_save_dir, 'result_log.csv')):
            result_logs = pd.read_csv(os.path.join(self.result_save_dir, 'result_log.csv'))
            result_logs = result_logs.append(cur_log, ignore_index=True)
        else:
            result_logs = cur_log
        result_logs = result_logs.sort(['ip'])
        result_logs.to_csv(os.path.join(self.result_save_dir, "result_log.csv"), index=False)

    def logging_lerp_avg(self, result_avg_log, lerp_name, verbose=False):
        self.lerp_avg_logs.append([lerp_name] +
                         result_avg_log)
        if verbose:
            print("psnr_avg:{:0.3f}, ssim_avg:{:0.3f}, nrmse_avg:{:0.3f}"
                  .format(result_avg_log[0], result_avg_log[1], result_avg_log[2]))

    def save_lerp_result_avg_log(self):
        self.result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(self.args.test_dir.split('/')[-3:]), 'diff_model', 'with_truth')
        
        if not os.path.exists(self.result_save_dir):
            os.makedir(self.result_save_dir)
        cur_log = pd.DataFrame(self.lerp_avg_logs,
                                 columns=self.avg_headers)

        if os.path.exists(os.path.join(self.result_save_dir, 'result_avg_log.csv')):
            result_logs = pd.read_csv(os.path.join(self.result_save_dir, 'result_avg_log.csv'))
            result_logs = result_logs.append(cur_log, ignore_index=True)
        else:
            result_logs = cur_log
        result_logs = result_logs.sort(['model'])
        result_logs.to_csv(os.path.join(self.result_save_dir, "result_avg_log.csv"), index=False)

class Tester(Base):
    def testing_model(self, model_name, test_dir):
        args = self.args
        model = load_checkpoint(model_name)
        
        print('===> Testing')
        if args.ground_truth:
            self.evaluating_model(model, model_name, test_dir)
        else:
            result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(args.test_dir.split('/')[-3:]), 'diff_model', 'without_truth')
            if not os.path.exists(result_save_dir):
                os.makedirs(result_save_dir)
        
            img_files = sorted(os.listdir(test_dir))
            for img_file in img_files:    
                img_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(test_dir, img_file)
                _model_name = os.path.splitext(model_name)[0]
                output_file = img_name + '_' + _model_name + os.path.splitext(img_file)[1]
                output_path = os.path.join(result_save_dir, output_file)
                result_generator = Result_Generator_Without_Truth(args, img_path, model)
                out_img = result_generator()
                out_img.save(output_path)
                
                lerp_file = img_name + '_up' + str(args.upscale_factor) + '_lerp' + os.path.splitext(img_file)[1]
                lerp_path = os.path.join(result_save_dir, lerp_file)
                if not os.path.exists(lerp_path):
                    lr_img = Image.open(img_path)
                    lerp_img = lr_img.resize((lr_img.size[0] * args.upscale_factor, lr_img.size[1] * args.upscale_factor), resample=Image.BICUBIC)
                    lerp_img.save(lerp_path)
                                
    def testing_middle_checkpoint(self, model_name, test_dir):
        args = self.args
        checkpoint_dir = os.path.splitext(model_name)[0]
        self.checkpoint_dir = checkpoint_dir
        checkpoint_path = os.path.join(DIR, 'model_zoo', checkpoint_dir)
        
        print('===> Testing')
        
        for checkpoint_name in os.listdir(checkpoint_path):
            model =  load_middle_checkpoint(checkpoint_dir, checkpoint_name)  
            if args.ground_truth:    
                self.evaluating_middle_checkpoint(model, checkpoint_name, checkpoint_dir, test_dir)
            else:
                result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(args.test_dir.split('/')[-3:]), 'middle_checkpoint',checkpoint_dir,'without_truth')
                if not os.path.exists(result_save_dir):
                    os.makedirs(result_save_dir)
                img_files = sorted(os.listdir(test_dir))    
                for img_file in img_files:    
                    img_name = os.path.splitext(img_file)[0]
                    img_path = os.path.join(args.test_dir, img_file)
                    _checkpoint_name = os.path.splitext(checkpoint_name)[0]
                    output_file = img_name + '_' + _checkpoint_name + os.path.splitext(img_file)[1]
                    output_path = os.path.join(result_save_dir, output_file)
                    result_generator = Result_Generator_Without_Truth(args, img_path, model)
                    out_img = result_generator()
                    out_img.save(output_path)
        
    def evaluating_model(self, model, model_name, data_dir):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
        """
        metrics
        """
#        psnr, nrmse, ssim, vifp, fsim
        psnr_all, nrmse_all, ssim_all = 0, 0, 0
        psnr_lerp_all, nrmse_lerp_all, ssim_lerp_all = 0, 0, 0
        flag = 0
        model.eval()
        
        result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(args.test_dir.split('/')[-3:]), 'diff_model', 'with_truth')
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)
        
        img_files = sorted(os.listdir(data_dir))
        
        for img_file in img_files:
            img_name = os.path.splitext(img_file)[0]  
            img_path = os.path.join(args.test_dir, img_file)
            _model_name = os.path.splitext(model_name)[0]
            output_file = img_name + '_' + _model_name + os.path.splitext(img_file)[1]
            output_path = os.path.join(result_save_dir, output_file)
            
            result_generator = Result_Generator_With_Truth(args, img_path, model)
            sr_img, (psnr, ssim, nrmse) = result_generator()
            """
            metrics
            """
            psnr_all += psnr
            ssim_all += ssim
            nrmse_all += nrmse
            
            self.ip = img_file

            self.result_log = [round(idx, 3) for idx in [psnr, nrmse, ssim]]
            self.logging(self.result_log)
            
            # generate lr image
            lr_file = img_name + '_up' + str(args.upscale_factor) +'_lr'+ os.path.splitext(img_file)[1]
            lr_lerp_file = img_name + '_up' + str(args.upscale_factor) +'_lr_lerp' + os.path.splitext(img_file)[1]
            lr_path = os.path.join(result_save_dir, lr_file)
            lr_lerp_path = os.path.join(result_save_dir, lr_lerp_file)
            if not os.path.exists(lr_path):
                hr_img = Image.open(img_path)
                lr_img = hr_img.resize((hr_img.size[0]// args.upscale_factor, hr_img.size[1] // args.upscale_factor))
                lr_img.save(lr_path)
            if not os.path.exists(lr_lerp_path):
                flag += 1
                hr_img = Image.open(img_path)
                lr_img = Image.open(lr_path)
                lerp_img = lr_img.resize((lr_img.size[0] * args.upscale_factor, lr_img.size[1] * args.upscale_factor), resample=Image.BICUBIC)
                lerp_img.save(lr_lerp_path)
                psnr_lerp, nrmse_lerp, ssim_lerp = lerp_img_evaluation(hr_img, lerp_img)
                lerp_name = 'up' + str(args.upscale_factor) + '_lr_lerp'
                
                result_log_lerp = [round(idx, 3) for idx in [psnr_lerp, nrmse_lerp, ssim_lerp]]
                self.lerp_logging(result_log_lerp, lerp_name)
                self.save_lerp_result_log()
                self.lerp_logs = []
                psnr_lerp_all += psnr_lerp
                nrmse_lerp_all += nrmse_lerp
                ssim_lerp_all += ssim_lerp
                
            sr_img.save(output_path)
        self.save_result_log()

        psnr_avg, ssim_avg, nrmse_avg   = psnr_all / len(img_files), ssim_all / len(img_files), nrmse_all / len(img_files)         
        self.result_avg_log = [round(idx, 3) for idx in [psnr_avg, ssim_avg, nrmse_avg]]        
        self.logging_avg(self.result_avg_log)
        self.save_result_avg_log()
        
        if flag == len(img_files):
            psnr_lerp_avg, ssim_lerp_avg, nrmse_lerp_avg   = psnr_lerp_all / len(img_files), ssim_lerp_all / len(img_files), nrmse_lerp_all / len(img_files) 
            self.lerp_result_avg_log = [round(idx, 3) for idx in [psnr_lerp_avg, ssim_lerp_avg, nrmse_lerp_avg]]
            self.logging_lerp_avg(self.lerp_result_avg_log, lerp_name)
            self.save_lerp_result_avg_log()
            self.lerp_avg_logs = []
            flag = 0



    def evaluating_middle_checkpoint(self, model, checkpoint_name, checkpoint_dir, data_dir):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
        """
        metrics
        """
#        psnr, nrmse, ssim, vifp, fsim
        psnr_all, nrmse_all, ssim_all = 0, 0, 0
        model.eval()
        
        result_save_dir = os.path.join(Result_DIR, 'raw', '_'.join(args.test_dir.split('/')[-3:]), 'middle_checkpoint',checkpoint_dir,'with_truth')
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)
        
        img_files = sorted(os.listdir(data_dir))
        
        for img_file in img_files:
            img_name = os.path.splitext(img_file)[0]  
            img_path = os.path.join(args.test_dir, img_file)
            _checkpoint_name = os.path.splitext(checkpoint_name)[0]
            output_file = img_name + '_' + _checkpoint_name + os.path.splitext(img_file)[1]
            output_path = os.path.join(result_save_dir, output_file)
            
            result_generator = Result_Generator_With_Truth(args, img_path, model)
            sr_img, (psnr, ssim, nrmse) = result_generator()
            sr_img.save(output_path)
            
            """
            metrics
            """
            psnr_all += psnr
            ssim_all += ssim
            nrmse_all += nrmse
            
            self.ip = img_file
            self.result_log = [round(idx, 3) for idx in [psnr, nrmse, ssim]]
            self.middle_logging(self.result_log, _checkpoint_name)
            
            # generate lr and linear interpolation image
            lr_file = img_name + '_up' + str(args.upscale_factor) + '_lr' + os.path.splitext(img_file)[1]
            lr_lerp_file = img_name + '_up' + str(args.upscale_factor) + '_lr_lerp' + os.path.splitext(img_file)[1]
            lr_path = os.path.join(result_save_dir, lr_file)
            lr_lerp_path = os.path.join(result_save_dir, lr_lerp_file)
            if not os.path.exists(lr_path):
                hr_img = Image.open(img_path)
                lr_img = hr_img.resize((hr_img.size[0]// args.upscale_factor, hr_img.size[1] // args.upscale_factor))
                lr_img.save(lr_path)
            if not os.path.exists(lr_lerp_path):
                hr_img = Image.open(img_path)
                lr_img = Image.open(lr_path)
                lerp_img = lr_img.resize((lr_img.size[0] * args.upscale_factor, lr_img.size[1] * args.upscale_factor), resample=Image.BICUBIC)
                lerp_img.save(lr_lerp_path)
                
            sr_img.save(output_path)
        self.save_middle_result_log()
        psnr_avg, ssim_avg, nrmse_avg   = psnr_all / len(img_files), ssim_all / len(img_files), nrmse_all / len(img_files) 
        self.result_avg_log = [round(idx, 3) for idx in [psnr_avg, ssim_avg, nrmse_avg]]
        self.logging_middle_avg(self.result_avg_log, _checkpoint_name)
        self.save_middle_result_avg_log()
        self.middle_logs = []
        self.middle_avg_logs = []
        