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

Utils_DIR = os.path.dirname(os.path.abspath(__file__))
Logs_DIR = os.path.join(Utils_DIR, '../logs')
Checkpoint_DIR = os.path.join(Utils_DIR, '../model_zoo')

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def load_checkpoint(name):
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, name)
                          ), "{} not exists.".format(name)
    print("Loading model: {}".format(name))
    return torch.load("{}/{}".format(Checkpoint_DIR, name))

class Base(object):
    def __init__(self, args, method):
        self.args = args
        self.method = method
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.date = time.strftime("%h%d_%H")
        self.epoch = 0
        self.iter = 0
        self.logs = []
        self.headers = ["epoch", "iter", "train_loss", "train_psnr", "train_time(sec)", "train_fps", "val_loss", "val_psnr", "val_time(sec)", "val_fps"]

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

        self.logs.to_csv("{}/raw/{}_{}_{}_{}.csv".format(Logs_DIR, self.method, self.args.trigger,
                                                      self.args.nEpochs, self.date), index=False, float_format='%.3f')

    def save_checkpoint(self, model, name=None):
        if self.args.cuda:
            model.cpu()
        if name:
            model_name = "{}_{}_{}_{}_{}.pth".format(
                self.method, name, self.args.trigger, self.args.nEpochs, self.date)
        model_name = "{}_{}_{}_{}.pth".format(
            self.method, self.args.trigger, self.args.nEpochs, self.date)
        if not os.path.exists(Checkpoint_DIR):
            os.mkdir(Checkpoint_DIR)
        torch.save(model, os.path.join(Checkpoint_DIR, model_name))
        print("===> Saving checkpoint: {}".format(model_name))
        
        self.save_args(model_name)

    def save_args(self, model_name):
        args = self.args
        args_info_path = os.path.join(Checkpoint_DIR, 'model_info.txt')
        with open(args_info_path, 'a') as f:
            f.write(model_name + ': \n')
            f.write('\t' + str(args) + ': \n') 
            f.close()
        
        
    def learning_curve(self, labels=["train_loss", "train_psnr", "val_loss", "val_psnr"]):
        if not os.path.exists(os.path.join(Logs_DIR, "curve")):
            os.mkdir(os.path.join(Logs_DIR, "curve"))
        # set style
        sns.set_context("paper", font_scale=1.5,)
        sns.set_style("ticks", {
            "font.family": "Times New Roman",
            "font.serif": ["Times", "Palatino", "serif"]})
        
        for _label in labels:
            
            plt.plot(self.logs[self.args.trigger],
                     self.logs[_label], label=_label)
        plt.ylabel("Loss / PSNR")
        if self.args.trigger == 'epoch':
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.suptitle("Training log of {}".format(self.method))
        # remove top&left line
        # sns.despine()
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.savefig(os.path.join(Logs_DIR, 'curve', '{}_{}_{}_{}.png'.format(self.method, self.args.trigger, self.args.nEpochs, self.date)),
                    format='png', bbox_inches='tight', dpi=1200)
        #plt.savefig('curve/{}_curve.eps'.format(fig_title), format='eps', bbox_inches='tight', dpi=1200)

class Trainer(Base):
    """
    args.iters: total iterations for all epochs
    steps: iteration per epoch 
    """
    def training(self, net, datasets):
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
                                     num_workers=args.threads, shuffle=True)
            batch_iterator = iter(data_loader)
            epoch_loss, epoch_psnr = 0, 0
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
                epoch_loss += loss.item()
                epoch_psnr += metrics.psnr(gen_y.data, y.data)

#                print("===> Epoch[{}]({}/{}): Loss: {:.4f}; \t PSNR: {:.4f}"
#                      .format(epoch, step+1, steps, loss.item(), metrics.psnr(gen_y.data, y.data)))
                
                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [loss.item(), metrics.psnr(gen_y.data, y.data), _time, nb_samples / _time]
#                    train_log = [log_loss / args.iter_interval, log_psnr /
#                                 args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
#                    log_loss, log_psnr = 0, 0
            print("===> Epoch {} Complete: Avg. Loss: {:.4f}; \t Avg. PSNR: {:.4f}"
                  .format(epoch, epoch_loss / steps, epoch_psnr / steps))
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
        val_loss, val_psnr = 0, 0
        data_loader = DataLoader(dataset=dataset, batch_size=args.testbatch_size, num_workers=args.threads,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.testbatch_size
#        model.eval()
        start = time.time()
        for step in range(steps):
            x, y = next(batch_iterator)

            x = x.to(self.device)
            y = y.to(self.device)
 
            # calculate pixel accuracy of generator
            gen_y = model(x)

            val_loss += F.mse_loss(gen_y, y).item()
            val_psnr += metrics.psnr(gen_y.data, y.data)
#            print(metrics.psnr(gen_y.data, y.data))

        _time = time.time() - start
        nb_samples = steps * args.batch_size
        val_log = [val_loss / steps, val_psnr /
                   steps, _time, nb_samples / _time]
        self.val_log = [round(x, 3) for x in val_log]

    def evaluating(self, model, dataset, split):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
#        oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
        psnr = 0
        model.eval()
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.batch_size

        start = time.time()
        for step in range(steps):
            x, y = next(batch_iterator)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
            # calculate pixel accuracy of generator
            gen_y = model(x)
            psnr += metrics.psnr(gen_y.data, y.data)

        _time = time.time() - start

        if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
            os.makedirs(os.path.join(Logs_DIR, 'statistic'))

        # recording performance of the model
        nb_samples = steps * args.batch_size
        basic_info = [self.date, self.method,
                      self.epoch, self.iter, nb_samples, _time]
        basic_info_names = ['date', 'method', 'epochs',
                            'iters', 'nb_samples', 'time(sec)']

        perform = [round(idx / steps, 3)
                   for idx in [psnr]]
        perform_names = ["psnr"]
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
