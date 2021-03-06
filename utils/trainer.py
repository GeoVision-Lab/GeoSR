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

Utils_DIR = os.path.dirname(os.path.abspath(__file__))
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


class Base(object):
    def __init__(self, args, method):
        self.args = args
        self.method = method
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.date = time.strftime("%h%d_%H")
        self.epoch = 0
        self.iter = 0
        self.logs = []
        """
        metrics
        psnr, nrmse, ssim, vifp, fsim
        """
        self.headers = ["epoch", "iter", "train_loss", "train_psnr", "train_ssim", "train_nrmse", "train_time(sec)", "train_fps",\
                        "val_loss", "val_psnr", "val_ssim", "val_nrmse", "val_time(sec)", "val_fps", ]

    def logging(self, verbose=True):
        self.logs.append([self.epoch, self.iter] +
                         self.train_log + self.val_log)
        if verbose:
            print("Epoch:{:02d}, Iter:{:05d}, train_loss:{:0.3f}, train_psnr:{:0.3f}, val_loss:{:0.3f}, val_psnr:{:0.3f}."
                  .format(self.epoch, self.iter, self.train_log[0], self.train_log[1], self.val_log[0], self.val_log[1]))

    def save_log(self):
        if not os.path.exists(os.path.join(Logs_DIR, 'raw')):
            os.makedirs(os.path.join(Logs_DIR, 'raw'))
        self.logs = pd.DataFrame(self.logs,
                                 columns=self.headers)

        self.logs.to_csv("{}/raw/up{}_{}_{}_{}_{}.csv".format(Logs_DIR, self.args.upscale_factor, self.method, self.args.trigger,
                                                      self.args.nEpochs, self.date), index=False, float_format='%.3f')

    def save_checkpoint(self, model, name=None):
        if self.args.cuda:
            model.cpu()
        if name:
            model_name = "up{}_{}_{}_{}_{}_{}.pth".format(
                self.args.upscale_factor, self.method, name, self.args.trigger, self.args.nEpochs, self.date)
        model_name = "up{}_{}_{}_{}_{}.pth".format(
            self.args.upscale_factor, self.method, self.args.trigger, self.args.nEpochs, self.date)
        if not os.path.exists(Checkpoint_DIR):
            os.mkdir(Checkpoint_DIR)
        torch.save(model, os.path.join(Checkpoint_DIR, model_name))
        print("===> Saving checkpoint: {}".format(model_name))
        self.save_model_info(model_name)
        return model_name
    
    def save_middle_checkpoint(self, model, epoch, iteration, model_name, name=None):
        """
        save middle checkpoint of model named model_name in file model_name 
        """
        model_epoch = copy.deepcopy(model)
        if self.args.cuda:
            model_epoch.cpu()
        if name:
            model_epoch_name = "{}_epoch_{}_iter_{}.pth".format(
                name, epoch, iteration)
        model_epoch_name = "epoch_{}_iter_{}.pth".format(
            epoch, iteration)
        Checkpoint_Epoch_DIR = os.path.join(Checkpoint_DIR, model_name)
        if not os.path.exists(Checkpoint_Epoch_DIR):
            os.mkdir(Checkpoint_Epoch_DIR)
        torch.save(model_epoch, os.path.join(Checkpoint_Epoch_DIR, model_epoch_name))
    
    def save_model_info(self, model_name):
        args = self.args
        model_info_names, model_info = [], []
        for name in vars(args):
            model_info_names.append(name)
            model_info.append(eval('args.'+str(name)))
        basic_info_names = ['date', 'method', 'model_name']
        basic_info = [self.date, self.method, model_name]
        model_log = pd.DataFrame([basic_info + model_info],
                               columns=basic_info_names + model_info_names)

        if os.path.exists(os.path.join(Logs_DIR, 'statistic', 'model_info.csv')):
            logs = pd.read_csv(os.path.join(
                Logs_DIR, 'statistic', 'model_info.csv'))
        else:
            logs = pd.DataFrame([])
        logs = logs.append(model_log, ignore_index=True)
        logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                 'model_info.csv'), index=False)
        
        
    def learning_curve(self, labels=["train_loss", "train_psnr", "val_loss", "val_psnr"], trans = True):
        if not os.path.exists(os.path.join(Logs_DIR, "curve")):
            os.mkdir(os.path.join(Logs_DIR, "curve"))
        # set style
        sns.set_context("paper", font_scale=1.5,)
        sns.set_style("ticks", {
            "font.family": "Times New Roman",
            "font.serif": ["Times", "Palatino", "serif"]})
        if trans:
            # for better visualization
            plt.plot(self.logs[self.args.trigger],
                     self.logs[labels[0]], label=labels[0])        
            plt.plot(self.logs[self.args.trigger],
                     self.logs[labels[1]]/100, label=labels[1])
            plt.plot(self.logs[self.args.trigger],
                     self.logs[labels[2]], label=labels[2])
            plt.plot(self.logs[self.args.trigger],
                     self.logs[labels[3]]/100, label=labels[3])
        else:
            for _label in labels:
                plt.plot(self.logs[self.args.trigger],
                         self.logs[_label], label=_label)

        plt.ylabel("Loss & PSNR/100")
        if self.args.trigger == 'epoch':
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.suptitle("Training log of {}".format(self.method))
        # remove top&left line
        # sns.despine()
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.savefig(os.path.join(Logs_DIR, 'curve', 'up{}_{}_{}_{}_{}.png'.format(self.args.upscale_factor, self.method, self.args.trigger, self.args.nEpochs, self.date)),
                    format='png', bbox_inches='tight', dpi=1200)
        #plt.savefig('curve/{}_curve.eps'.format(fig_title), format='eps', bbox_inches='tight', dpi=1200)

class Trainer(Base):
    """
    args.iters: total iterations for all epochs
    steps: iteration per epoch 
    """
    def training(self, net, datasets, verbose=False):
        """
          input:
            net: (object) model & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        steps = len(datasets[0]) // args.batch_size

        if args.trigger == 'epoch':
            args.epochs = args.nEpochs
            args.iters = steps * args.nEpochs
            args.iter_interval = steps * args.interval
        else:
            args.iters = args.nEpochs
            args.epochs = args.nEpochs // steps + 1
            args.iter_interval = args.interval

        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(dataset=datasets[0], batch_size=args.batch_size, 
                                     num_workers=args.threads, shuffle=False)
            batch_iterator = iter(data_loader)
            """
            metrics
            """
            epoch_loss, epoch_psnr= 0, 0
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                x, y = next(batch_iterator)
                x = x.to(self.device)
                y = y.to(self.device)
                # training
                gen_y = net(x)
                loss = F.mse_loss(gen_y, y)
                # Update generator parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                """
                metrics
                """
                epoch_loss += loss.item()
                epoch_psnr += metrics.psnr(gen_y.data, y.data)
#                epoch_nrmse += metrics.nrmse(gen_y.data, y.data)
#                epoch_ssim += metrics.ssim(gen_y.data, y.data)
#                epoch_vifp += metrics.vifp(gen_y.data, y.data)
                if verbose:
                    print("===> Epoch[{}]({}/{}): Loss: {:.4f}; \t PSNR: {:.4f}"
                          .format(epoch, step+1, steps, loss.item(), metrics.psnr(gen_y.data, y.data)))
                    
                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    """
                    metrics
                    """
                    loss_log = loss.item()
                    psnr_log = metrics.psnr(gen_y.data, y.data)
                    nrmse_log = metrics.nrmse(gen_y.data, y.data)
                    ssim_log = metrics.ssim(gen_y.data, y.data)
#                    vifp_log = metrics.ssim(gen_y.data, y.data)                    
                    train_log = [loss_log, psnr_log, nrmse_log, ssim_log, _time, nb_samples / _time]
#                    train_log = [log_loss / args.iter_interval, log_psnr /
#                                 args.iter_interval, _time, nb_samples / _time]
                    
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.args.middle_checkpoint:
                        model_name_dir = "up{}_{}_{}_{}_{}".format(
                                        self.args.upscale_factor, self.method, self.args.trigger, self.args.nEpochs, self.date)
                        self.save_middle_checkpoint(net, self.epoch, self.iter, model_name_dir)
                        
                    # reinitialize
                    start = time.time()
#                    log_loss, log_psnr = 0, 0
            print("===> Epoch {} Complete: Avg. Loss: {:.4f}; \t Avg. PSNR: {:.4f}"
                  .format(epoch, epoch_loss / steps, epoch_psnr / steps))
            """
            metrics
            """
            epoch_loss, epoch_psnr = 0, 0 

    def validating(self, model, dataset):
        """
          input:
            model: (object) pytorch model
            batch_size: (int)
            dataset : (object) dataset
          return [val_mse, val_loss]
        """
        args = self.args
        """
        metrics
        """
        val_loss, val_psnr, val_nrmse, val_ssim = 0, 0, 0, 0
        data_loader = DataLoader(dataset=dataset, batch_size=args.valbatch_size, num_workers=args.threads,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.valbatch_size
#        model.eval()
        start = time.time()
        for step in range(steps):
            x, y = next(batch_iterator)
            x = x.to(self.device)
            y = y.to(self.device) 
            # calculate pixel accuracy of generator
            gen_y = model(x)
            """
            metrics
            """
            val_loss += F.mse_loss(gen_y, y).item()
            val_psnr += metrics.psnr(gen_y.data, y.data)
            val_nrmse += metrics.nrmse(gen_y.data, y.data)
            val_ssim += metrics.ssim(gen_y.data, y.data)
#            val_vifp += metrics.vifp(gen_y.data, y.data)

        _time = time.time() - start
        nb_samples = steps * args.valbatch_size
        """
        metrics
        """
        val_log = [val_loss / steps, val_psnr /
                   steps, val_nrmse/steps, val_ssim/steps, _time, nb_samples / _time]
        self.val_log = [round(x, 3) for x in val_log]

    def evaluating(self, model, dataset, split):
        """
        Evaluate overall performance of the model
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
        psnr, nrmse, ssim = 0, 0, 0
        model.eval()
        data_loader = DataLoader(dataset, args.evalbatch_size, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.evalbatch_size

        start = time.time()
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
            psnr += metrics.psnr(gen_y, y)
            nrmse += metrics.nrmse(gen_y, y)
            ssim += metrics.ssim(gen_y, y)
#            vifp += metrics.vifp(gen_y.data, y.data)
        _time = time.time() - start

        if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
            os.makedirs(os.path.join(Logs_DIR, 'statistic'))

        # recording performance of the model
        nb_samples = steps * args.evalbatch_size
        fps = nb_samples / _time
        basic_info = [self.date, self.method,
                      self.epoch, self.iter, nb_samples, _time, fps]
        basic_info_names = ['date', 'method', 'epochs',
                            'iters', 'nb_samples', 'time(sec)', 'fps']        
        """
        metrics
        """
        perform = [round(idx / steps, 3)
                   for idx in [psnr, nrmse, ssim]]
        perform_names = ['psnr', 'nrmse', 'ssim']
        cur_log = pd.DataFrame([basic_info + perform],
                               columns=basic_info_names + perform_names)
        # save performance
        if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}.csv".format(split))):
            logs = pd.read_csv(os.path.join(
                Logs_DIR, 'statistic', "{}.csv".format(split)))
        else:
            logs = pd.DataFrame([])
        logs = logs.append(cur_log, ignore_index=True)
        logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                 "{}.csv".format(split)), index=False, float_format='%.3f')


