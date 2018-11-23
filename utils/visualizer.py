#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:28:24 2018

@author: kaku
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import datetime

Utils_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(Utils_DIR, '..')
Logs_DIR = os.path.join(Utils_DIR, '../logs')
Checkpoint_DIR = os.path.join(Utils_DIR, '../model_zoo')
Result_DIR = os.path.join(Utils_DIR, '../result')

def sub_enlarge(img, center, size, block_width = 2):
    """
    Enlarge sub regions and draw the block
    Parameters
    ----------
        img: numpy.ndarray
        center: make left uper point to be the center
        size: sub rigion size
        block_width: width of block line
    """
    color_disk = {"firebrick" : [178, 34, 34], 
                  'limegreen' : [50, 205, 50]}
    color = color_disk['firebrick']
    
    img_rows, img_cols = img.shape[0], img.shape[1]
    if center[0]+size >= img_rows//2 or center[1]+size >= img_cols//2:
        print('overlapped, please select a better sub region to enlarge')
    
    sub_img = img[center[0]:center[0]+size, center[1]:center[1]+size]
    enlarged_sub_img = cv2.resize(sub_img, (img_rows//2, img_cols//2))
    
    img[img_rows//2:, img_cols//2:] = enlarged_sub_img
    
    # draw line for enlarged part
    img[(img_rows//2):(img_rows//2+block_width), img_cols//2:] = color
    img[(img_rows//2):, (img_cols//2):(img_cols//2+block_width)] = color
    img[-block_width:, img_cols//2:] = color
    img[img_rows//2:, -block_width:] = color
    
    # draw line for sub rigion
    img[center[0]: center[0]+block_width, center[1]:center[1]+size] = color
    img[center[0]:center[0]+size, center[1]:center[1]+block_width] = color
    img[center[0]+size-block_width : center[0]+size, center[1]:center[1]+size] = color
    img[center[0]: center[0]+size, center[1]+size-block_width:center[1]+size] = color        
    return img

def model_titles_change(model_names, trans=True):
    if trans:
        new_model_names = []
        for i in range(len(model_names)):
            model_name = model_names[i]
            if model_name == 'HR':
                new_model_name = model_name
            else:
                model_name_split = model_name.split('_')
                if 'lerp' in model_name_split:
                    new_model_name = 'BICUBIC' 
                else:
                    upscale_factor_index = ['up' in model_name_split[i] for i in range(len(model_name_split))].index(True)                    
                    new_model_name = model_name_split[upscale_factor_index + 1]
            new_model_names.append(new_model_name.upper())
    else:
        new_model_names = model_names
    return new_model_names

def image_titles_change(image_names, trans=True):
    if trans:
        new_image_names = []
        for i in range(len(image_names)):
            new_image_name =  '('+chr(97 + i)+')'
            new_image_names.append(new_image_name)
    else:
        new_image_names = image_names
    return new_image_names

def get_iter_info(img_names, result_dir):
    """
    Return Example
    ------
    
    """
    _sample_img_name = os.path.splitext(img_names[0])[0]
    img_names = os.listdir(result_dir)
    bool_list = [_sample_img_name in img_names[i] for i in range(len(img_names))]
    pop_num = bool_list.count(True)
    iter_img_names = []
    epochs = []
    iters = []
    for i in range(pop_num):
        pop_idx = bool_list.index(True)
        iter_img_name = img_names[pop_idx]
        bool_list[pop_idx]=False
        if 'lr' not in iter_img_name:
#            print(iter_img_name)
            _iter_img_name = os.path.splitext(iter_img_name)[0]
            iter_img_name_split = _iter_img_name.split('_')
            epoch_idx = iter_img_name_split.index('epoch')
            iter_idx = iter_img_name_split.index('iter')
            _epoch = iter_img_name_split[epoch_idx + 1]
            _iter = iter_img_name_split[iter_idx + 1]
            epochs.append(int(_epoch))
            iters.append(int(_iter))
            iter_img_names.append(iter_img_name)
            epochs = sorted(epochs)
            iters = sorted(iters)
    return epochs, iters
    
    
class Figure_Compare(object):
    def __init__(self, args):
        self.args = args

    def model_compare(self, model_names, cand_num):
        args = self.args
        test_img_names = sorted(os.listdir(args.test_dir))
        cand_img_names = np.random.choice(test_img_names, cand_num, replace=False)
        
        new_model_names = model_titles_change(model_names, trans = True)
        new_image_names = image_titles_change(cand_img_names, trans = True)
        
        col_num = cand_num
        row_num = len(model_names)
    
        plt.figure(figsize = (col_num, row_num), dpi=args.dpi)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(col_num):
            for j in range(row_num):
                idx = col_num * j + i + 1
                plt.subplot(row_num, col_num, idx)
                # get rid of the the frame
                plt.box(False)
                if model_names[j] == 'HR':
                    img = plt.imread(os.path.join(args.test_dir, cand_img_names[i]))
                else:
                    cand_img_name = os.path.splitext(cand_img_names[i])[0]
                    ext = os.path.splitext(cand_img_names[i])[1]
                    img_name = cand_img_name + '_' + model_names[j] + ext
                    img = plt.imread(os.path.join(args.raw_result_dir_diff_model, img_name)) 
                if args.enlarge_sub_region:
                    center, size, width = args.sub_region
                    if 'lr' in model_names[j] and not 'lerp' in model_names[j]:
                        upscale_factor = int(model_names[j][(model_names[j].index('up')+2):(model_names[j].index('_'))])
                        img = sub_enlarge(img, (center[0]//upscale_factor,center[1]//upscale_factor), size//upscale_factor, block_width = width//upscale_factor)
                    else:
                        img = sub_enlarge(img, center, size, block_width = width)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.gca().set_title(new_image_names[i], fontsize=7)
                if i == 0:
                    plt.ylabel(new_model_names[j], fontsize=5, style = 'italic')
                    
                plt.imshow(img)
        plt.tight_layout()
        plt.savefig(os.path.join(args.gen_result_dir,'result_diff_model_'+args.time+'.png'),dpi=args.dpi) 
        
    def model_compare_single_image(self, model_names):
        args = self.args
        test_img_names = sorted(os.listdir(args.test_dir))
        cand_img_name = np.random.choice(test_img_names, 1, replace=False)[0]
        
        new_model_names = model_titles_change(model_names, trans = True)
        new_image_names = ['('+chr(97 + i)+')' for i in range(len(new_model_names))]
        
        col_num = len(new_model_names) // 2
        row_num = 2
    
        plt.figure(figsize = (col_num*2, row_num*2), dpi=args.dpi)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(col_num):
            for j in range(row_num):
                idx = col_num * j + i + 1
                plt.subplot(row_num, col_num, idx)
                # get rid of the the frame
                plt.box(False)
                if model_names[idx-1] == 'HR':
#                    print(cand_img_name)
                    img = plt.imread(os.path.join(args.test_dir, cand_img_name))
                else:
                    _cand_img_name = os.path.splitext(cand_img_name)[0]
                    ext = os.path.splitext(cand_img_name)[1]
                    img_name = _cand_img_name + '_' + model_names[idx-1] + ext
                    
                    img = plt.imread(os.path.join(args.raw_result_dir_diff_model, img_name)) 
                if args.enlarge_sub_region:
                    center, size, width = args.sub_region
                    if 'lr' in model_names[idx-1] and not 'lerp' in model_names[idx-1]:
                        upscale_factor = int(model_names[idx-1][(model_names[idx-1].index('up')+2):(model_names[idx-1].index('_'))])
                        img = sub_enlarge(img, (center[0]//upscale_factor,center[1]//upscale_factor), size//upscale_factor, block_width = width//upscale_factor)
                    else:
                        img = sub_enlarge(img, center, size, block_width = width)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(new_image_names[idx-1] + ': ' + new_model_names[idx-1], fontsize=10,)
                  
                plt.imshow(img)
        plt.tight_layout()
        plt.savefig(os.path.join(args.gen_result_dir,'result_diff_model_sigle_'+args.time+'.png'),dpi=args.dpi)
    
    def iter_compare(self, checkpoint_name, cand_num, label = 'iter'):
        args = self.args
        test_img_names = sorted(os.listdir(args.test_dir))
        cand_img_names = np.random.choice(test_img_names, cand_num, replace=False)
        iter_result_dir = os.path.join(DIR, 'result', 'raw', 'airplane-alloc_images_test', 'middle_checkpoint', checkpoint_name, 'with_truth') 
        
        epochs, iters = get_iter_info(cand_img_names, iter_result_dir)

        new_image_names = image_titles_change(cand_img_names, trans = True)
        
        col_num = cand_num
        row_num = len(epochs)
    
        plt.figure(figsize = (col_num, row_num), dpi=args.dpi)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(col_num):
            for j in range(row_num):
                idx = col_num * j + i + 1
                plt.subplot(row_num, col_num, idx)
                # get rid of the the frame
                plt.box(False)
                _cand_img_name = os.path.splitext(cand_img_names[i])[0]
                ext = os.path.splitext(cand_img_names[i])[1]
                img_name = _cand_img_name + '_epoch_' + str(epochs[j]) + '_iter_' + str(iters[j]) + ext 
                img = plt.imread(os.path.join(iter_result_dir, img_name)) 
                
                if args.enlarge_sub_region:
                    center, size, width = args.sub_region
                    img = sub_enlarge(img, center, size, block_width = width)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.gca().set_title(new_image_names[i], fontsize=7)
                if i == 0:
                    if label == 'iter':
                        plt.ylabel(iters[j], fontsize=5, style = 'italic')
                    else:
                        plt.ylabel(epochs[j], fontsize=5, style = 'italic')
                    
                plt.imshow(img)
        plt.tight_layout()
        plt.savefig(os.path.join(args.gen_result_dir,'result_diff_iter_'+args.time+'.png'),dpi=args.dpi) 

    def iter_compare_single_image(self, checkpoint_name, label = 'iter'):
        args = self.args
        test_img_names = sorted(os.listdir(args.test_dir))
        cand_img_name = np.random.choice(test_img_names, 1, replace=False)[0]
        iter_result_dir = os.path.join(DIR, 'result', 'raw', 'airplane-alloc_images_test', 'middle_checkpoint', checkpoint_name, 'with_truth') 
        
        epochs, iters = get_iter_info(cand_img_names, iter_result_dir)

        new_image_names = ['('+chr(97 + i)+')' for i in range(len(epochs))]
        
        col_num = len(epochs) // 2
        row_num = 2
    
        plt.figure(figsize = (col_num*2, row_num*2), dpi=args.dpi)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(col_num):
            for j in range(row_num):
                idx = col_num * j + i + 1
                plt.subplot(row_num, col_num, idx)
                # get rid of the the frame
                plt.box(False)
                _cand_img_name = os.path.splitext(cand_img_name)[0]
                ext = os.path.splitext(cand_img_name)[1]
                img_name = _cand_img_name + '_epoch_' + str(epochs[idx-1]) + '_iter_' + str(iters[idx-1]) + ext 
                img = plt.imread(os.path.join(iter_result_dir, img_name)) 
                
                if args.enlarge_sub_region:
                    center, size, width = args.sub_region
                    img = sub_enlarge(img, center, size, block_width = width)
                plt.xticks([])
                plt.yticks([])
                if label == 'iter':
                    plt.xlabel(new_image_names[idx-1] + ': ' + str(iters[idx-1]), fontsize=10,)
                else:
                    plt.xlabel(new_image_names[idx-1] + ': ' + str(epochs[idx-1]), fontsize=10,)
                
                plt.imshow(img)
        plt.tight_layout()
        plt.savefig(os.path.join(args.gen_result_dir,'result_diff_iter_single_'+args.time+'.png'),dpi=args.dpi)        

    def upper_compare_single(self, model_names):
        args = self.args
        test_img_names = sorted(os.listdir(args.test_dir))
        cand_img_name = np.random.choice(test_img_names, 1, replace=False)
        
        new_model_names = model_titles_change(model_names, trans = True)
        new_image_names = image_titles_change(cand_img_name, trans = True)
        
        col_num = len(model_names) // 2 
        row_num = 2
        
        plt.figure(figsize = (col_num, row_num), dpi=args.dpi)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(col_num):
            for j in range(row_num):
                idx = col_num * j + i + 1
                plt.subplot(row_num, col_num, idx)
                # get rid of the the frame
                plt.box(False)
                if model_names[j] == 'HR':
                    img = plt.imread(os.path.join(args.test_dir, cand_img_names[i]))
                else:
                    _cand_img_name = os.path.splitext(cand_img_names[i])[0]
                    ext = os.path.splitext(cand_img_names[i])[1]
                    img_name = _cand_img_name + '_' + model_names[j] + ext
                    img = plt.imread(os.path.join(args.raw_result_dir_diff_model, img_name)) 
                    upscale_factor = int(model_names[j][(model_names[j].index('up')+2):(model_names[j].index('_'))])
                if args.enlarge_sub_region:
                    center, size, width = args.sub_region
                    if 'lr' in model_names[j] and not 'lerp' in model_names[j]:
                        img = sub_enlarge(img, (center[0]//upscale_factor,center[1]//upscale_factor), size//upscale_factor, block_width = width//upscale_factor)
                    else:
                        img = sub_enlarge(img, center, size, block_width = width)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.gca().set_title(new_image_names[i], fontsize=7)
                if i == 0:
                    plt.ylabel(new_model_names[j]+'UP'+str(upscale_factor), fontsize=5, style = 'italic')
                    
                plt.imshow(img)
        plt.tight_layout()
        plt.savefig(os.path.join(args.gen_result_dir,'result_diff_upper_up'+str(upscale_factor)+'_'+args.time+'.png'),dpi=args.dpi) 

        
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Visualization tools')
    parser.add_argument('--test_dir', type=str, default=os.path.join(DIR, 'dataset','airplane-alloc','images', 'test'), help="testing data directory")
    parser.add_argument('--raw_result_dir_diff_model', type=str, default=os.path.join(DIR, 'result/raw/airplane-alloc_images_test/diff_model/with_truth/'), help="raw result directory")
#    parser.add_argument('--raw_result_dir_diff_iter', type=str, default=os.path.join(DIR, 'airplane-alloc_images_test/middle_checkpoint/up2_ESPCN_epoch_100_Nov16_22/with_truth/'), help="raw result directory")
    parser.add_argument('--gen_result_dir', type=str, default=os.path.join(DIR, 'result/generate/'), help="testing data directory")
    parser.add_argument('--dpi', type=int, default=500, help='save middle checkpoint of model or not')
    
    
    parser.add_argument('--middle_checkpoint', type=lambda x: (str(x).lower() == 'true'), default=False, help='save middle checkpoint of model or not')
    
    
    parser.add_argument('--enlarge_sub_region', type=lambda x: (str(x).lower() == 'true'), default=True, help='enlarge sub-regions')
    parser.add_argument('--ground_truth', type=lambda x: (str(x).lower() == 'true'), default=True, help='have ground truth or not')
    parser.add_argument('--test_model', type=lambda x: (str(x).lower() == 'true'), default=True, help='test different models')
    parser.add_argument('--test_middle_checkpoint', type=lambda x: (str(x).lower() == 'true'), default=False, help='have middle checkpoints of one model')
    
    parser.add_argument('--figure_compare', type=lambda x: (str(x).lower() == 'true'), default=True, help='figure of comparing')
    parser.add_argument('--figure_model_compare', type=lambda x: (str(x).lower() == 'true'), default=True, help='figure of comparing different models')
    parser.add_argument('--figure_iter_compare', type=lambda x: (str(x).lower() == 'true'), default=True, help='figure of comparing different iterations')
    parser.add_argument('--figure_upper_compare', type=lambda x: (str(x).lower() == 'true'), default=True, help='figure of comparing different upscale factor')
    parser.add_argument('--checkpoint_name', type=str, default='up2_ESPCN_epoch_200_Nov23_18', help='save middle checkpoint of model or not')
    args = parser.parse_args()
#    result = main(args)
    print(args)
    
    args.time = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%p")
    
    if args.enlarge_sub_region:
        center = (50,50)
        size = 60
        width = 4
        args.sub_region = center, size, width

    #np.random.seed(1)
    if args.figure_compare:
        figure_compare = Figure_Compare(args)
    
        if args.figure_model_compare:
            cand_num = 6
            model_names = ['HR', 'up2_lr_lerp', 'up2_SRCNN_epoch_100_Nov16_23', 'up4_ESPCN_epoch_100_Nov17_00', 'up2_ESPCN_epoch_100_Nov23_18', 'up2_VDSR_epoch_100_Nov16_23', 'up2_FSRCNN_epoch_100_Nov16_22', 'up2_SRDenseNet_epoch_100_Nov16_23']
            #model_names = ['HR', 'up2_lr', 'up2_lr_lerp', 'up2_ESPCN_epoch_30_Nov22_16', 'up2_ESPCN_epoch_200_Nov22_21'] 
            #model_names = ['HR', 'up2_lr', 'up2_ESPCN_epoch_200_Nov22_21'] 
            
            test_img_names = sorted(os.listdir(args.test_dir))
            cand_img_names = np.random.choice(test_img_names, cand_num, replace=False)
            
            new_model_names = model_titles_change(model_names, trans = True)
            new_image_names = image_titles_change(cand_img_names, trans = True)
            
            col_num = cand_num
            row_num = len(model_names)
                    
            figure_compare.model_compare(model_names, cand_num)
            figure_compare.model_compare_single_image(model_names)
        if args.figure_iter_compare:
            checkpoint_name = args.checkpoint_name
            cand_num = 6            
            figure_compare.iter_compare(checkpoint_name, cand_num, label='epoch')
            figure_compare.iter_compare_single_image(checkpoint_name, label='iter')
        if args.figure_upper_compare:
            cand_num = 6            
            model_names = ['up2_lr_lerp', 'up4_lr_lerp', 'up8_lr_lerp', 'up2_ESPCN_epoch_100_Nov17_00', 'up4_ESPCN_epoch_100_Nov17_00', 'up2_VDSR_epoch_100_Nov16_23', 'up8_ESPCN_epoch_100_Nov17_00']
#            figure_compare.upper_compare_single(model_names)
            
    