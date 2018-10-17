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
import shutil
import random
import itertools
import pandas as pd

#from skimage.io import imread, imsave
from matplotlib.pyplot import imread, imsave
import argparse

Utils_DIR = os.path.dirname(os.path.abspath(__file__))

class Processor_Save(object):
    """
    Save information and images
    methods
    -------
    save_infos: randomly split all-infos into train, val, and test
    save_slices: save slices in save_dir/folder
    """
    def __init__(self):
        print("Baic processor")

    def save_infos(self, df):
        all_file = os.path.join(self.save_dir, 'all-infos.csv')
        df.to_csv(all_file, index=False)
        nb_list = list(range(df.shape[0]))
        tv_edge = int(df.shape[0] * self.split[0])
        vt_edge = int(df.shape[0] * (1 - self.split[2]))
        # shuffle list
        random.seed(self.seed)
        random.shuffle(nb_list)
        train_df = df.iloc[nb_list[:tv_edge], :]
        train_df.to_csv(os.path.join(self.save_dir, 'train-infos.csv'), index=False)
        val_df = df.iloc[nb_list[tv_edge:vt_edge], :]
        val_df.to_csv(os.path.join(self.save_dir, 'val-infos.csv'), index=False)
        test_df = df.iloc[nb_list[vt_edge:], :]
        test_df.to_csv(os.path.join(self.save_dir, 'test-infos.csv'), index=False)

    def save_slices(self, img_slices, img_name, folder):
        if not os.path.exists(os.path.join(self.save_dir, folder)):
            os.mkdir(os.path.join(self.save_dir, folder))
        for i in range(len(img_slices)):
#            print(i)
#            print(img_slices[i].shape)
            imsave(os.path.join(self.save_dir, folder, img_name+"_{0}.png".format(i)),
                   img_slices[i])

class Processor(Processor_Save):
    def __init__(self, data_dir, img_rows, img_cols, nb_crop,
                 seed, stride=None):
        """
        data_dir: dir in ../data used to save original data
        """
        self.src_names = sorted(os.listdir(os.path.join(Utils_DIR, '../data', data_dir)))
        self.data_dir = data_dir

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.nb_crop = nb_crop
        self.split = [0.6, 0.2, 0.2]
        self.stride = stride if stride else img_rows
        self.seed = seed

    def extract_by_stride_slide(self):
        """
        Save sliced picecs in ../dataset/data_dir
        """
        # make save dirs
        self.save_dir = os.path.join(
            Utils_DIR, '../dataset', self.data_dir + '-str')
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        print("Processing via sliding window with stride")
        _statistics = []
        _infos = []
        for self.src_name in self.src_names:
            print("\t Image:{}/{}".format(self.src_names.index(self.src_name)+1, len(self.src_names)))
            self.src_path = os.path.join(Utils_DIR, '../data', self.data_dir, self.src_name)
            # extract slices from source
            self.src_img = imread(self.src_path)
            rows, cols = self.src_img.shape[:2]
            row_range = range(0, rows - self.img_rows, self.stride)
            col_range = range(0, cols - self.img_cols, self.stride)
            print("\t \t img_rows : {}; img_cols : {}".format(rows, cols))
            print("\t \t nb_rows : {}; nb_cols : {}".format(len(row_range), len(col_range)))
            print("\t \t nb_crop : {}".format(len(row_range) * len(col_range)))
            X_slices = []
            for i, j in itertools.product(row_range, col_range):
                img_src = self.src_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                X_slices.append(img_src)
    
            _statistic = [self.src_name, len(X_slices), len(row_range), len(col_range),
                          self.img_rows, self.img_cols]
            _statistics.append(_statistic)
            
            _info = [os.path.splitext(self.src_name)[0] + '_{}.png'.format(i) for i in range(len(X_slices))]
            _infos.extend(_info)
            # save slices
            self.save_slices(X_slices, os.path.splitext(self.src_name)[0], "image")            
            
        _file = os.path.join(self.save_dir, 'statistic.csv')
        pd.DataFrame(_statistics,
                     columns=["img_name","nb-samples", "nb_rows", "nb_cols", "img_rows", "img_cols"]).to_csv(_file, index=False)
        
        # save infos
        infos = pd.DataFrame(columns=['id'])
        
        infos['id'] = _infos
        self.save_infos(infos)

    def extract_by_random_slide(self):
        """
        Save sliced picecs in ../dataset/data_dir
        """
        # make save dirs
        self.save_dir = os.path.join(
            Utils_DIR, '../dataset', self.data_dir + '-rand')
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        print("Processing via sliding window with stride")
        _statistics = []
        _infos = []
        random.seed(self.seed)
        for self.src_name in self.src_names:
            print("\t Image:{}/{}".format(self.src_names.index(self.src_name)+1, len(self.src_names)))
            self.src_path = os.path.join(Utils_DIR, '../data', self.data_dir, self.src_name)
            # extract slices from source
            self.src_img = imread(self.src_path)
            rows, cols = self.src_img.shape[:2]
            print("\t \t img_rows : {}; img_cols : {}".format(rows, cols))
            print("\t \t nb_crop : {}".format(self.nb_crop))

            X_slices = []
            for _ in range(self.nb_crop):
                i = random.randint(0, rows - self.img_rows)
                j = random.randint(0, cols - self.img_cols)
                img_src = self.src_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                X_slices.append(img_src)
    
            _statistic = [self.src_name, len(X_slices), self.img_rows, self.img_cols]
            _statistics.append(_statistic)
            
            _info = [os.path.splitext(self.src_name)[0] + '_{}.png'.format(i) for i in range(len(X_slices))]
            _infos.extend(_info)
            # save slices
            self.save_slices(X_slices, os.path.splitext(self.src_name)[0], "image")            
            
        _file = os.path.join(self.save_dir, 'statistic.csv')
        pd.DataFrame(_statistics,
                     columns=["img_name","nb-samples", "img_rows", "img_cols"]).to_csv(_file, index=False)
        
        # save infos
        infos = pd.DataFrame(columns=['id'])
        
        infos['id'] = _infos
        self.save_infos(infos)
        
if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-data', type=str, default="map",
                        help='data dir for processing')
    parser.add_argument('-mode', type=str, default='slide-stride',
                        choices=['slide-stride', 'slide-rand'],
                        help='croping mode: slide-stride, slide-rand ')
    parser.add_argument('-img_rows', type=int, default=224,
                        help='img rows for croping ')
    parser.add_argument('-img_cols', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-nb_crop', type=int, default=400,
                        help='number of random crops ')
    parser.add_argument('-seed', type=int, default=123,
                        help='random seed to use. Default=123 ')
    parser.add_argument('-stride', type=int, default=224,
                        help='img cols for croping ')

    args = parser.parse_args()
    
    processor = Processor(args.data, args.img_rows, args.img_cols, args.nb_crop,
                                args.seed, args.stride, )        
    if args.mode == 'slide-stride':
        processor.extract_by_stride_slide()
    else:    
        processor.extract_by_random_slide()
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    