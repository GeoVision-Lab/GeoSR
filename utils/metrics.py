#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import sys
sys.path.append('./utils')

import torch
import numpy as np
#from datasets import *
from torch.utils.data import DataLoader
from math import log10
import pytorch_msssim
from pysptools import distance
import numpy
import scipy.signal
import scipy.ndimage

esp = 1e-5


def _binarize(y_data, threshold=0.5):
    """
    args:
        y_data : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return binarized y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data


def _get_tp(y_pred, y_true):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
    return true_positive
    """
    return torch.sum(y_true * y_pred)


def _get_fp(y_pred, y_true):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
    return false_positive
    """
    return torch.sum((1.0 - y_true) * y_pred)


def _get_tn(y_pred, y_true):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
    return true_negative
    """
    return torch.sum((1.0 - y_true) * (1.0 - y_pred))


def _get_fn(y_pred, y_true):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
    return false_negative
    """
    return torch.sum(y_true * (1.0 - y_pred))


def confusion_matrix(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return confusion matrix
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fp = _get_fp(y_pred, y_true)
    nb_tn = _get_tn(y_pred, y_true)
    nb_fn = _get_fn(y_pred, y_true)
    return [nb_tp, nb_fp, nb_tn, nb_fn]


def overall_accuracy(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return (tp+tn)/total
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp_tn = torch.sum(y_true == y_pred)
    return nb_tp_tn / (np.prod(y_true.shape))


def precision(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return tp/(tp+fp)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fp = _get_fp(y_pred, y_true)
    return nb_tp / (nb_tp + nb_fp + esp)


def recall(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return tp/(tp+fn)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fn = _get_fn(y_pred, y_true)
    return nb_tp / (nb_tp + nb_fn + esp)


def f1_score(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return 2*precision*recall/(precision+recall)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fp = _get_fp(y_pred, y_true)
    nb_fn = _get_fn(y_pred, y_true)
    _precision = nb_tp / (nb_tp + nb_fp + esp)
    _recall = nb_tp / (nb_tp + nb_fn + esp)
    return 2 * _precision * _recall / (_precision + _recall + esp)


def kappa(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return (Po-Pe)/(1-Pe)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fp = _get_fp(y_pred, y_true)
    nb_tn = _get_tn(y_pred, y_true)
    nb_fn = _get_fn(y_pred, y_true)
    nb_total = nb_tp + nb_fp + nb_tn + nb_fn
    Po = (nb_tp + nb_tn) / nb_total
    Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn) +
          (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
    return (Po - Pe) / (1 - Pe + esp)


def jaccard(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return intersection / (sum-intersection)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    _intersection = torch.sum(y_true * y_pred)
    _sum = torch.sum(y_true + y_pred)
    return _intersection / (_sum - _intersection + esp)


def mse(y_pred, y_true, threshold=None):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return mean_squared_error
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    return ((y_pred - y_true)**2).mean()


def psnr(y_pred, y_true, threshold=None):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return psnr
    """
    return 10 * log10(1 / mse(y_pred, y_true, threshold=None))

def nrmse(y_true, y_pred, norm_type='min-max'):
    
    """
    https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/simple_metrics.py#L46
    """
    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt(np.mean((y_pred*y_true), dtype=np.float64))
    elif norm_type == 'min-max':
        denom = y_true.max() - y_true.min()
    elif norm_type == 'mean':
        denom = y_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    denom_float = np.float(denom)
    mse_float = np.float(mse(y_true, y_pred))
    return np.sqrt(mse_float) / denom_float
#    return psnr(y_true, y_pred)

def ssim(y_true, y_pred):
    """
    https://github.com/jorge-pessoa/pytorch-msssim
    """
    m = pytorch_msssim.MSSSIM()
    ssim_tensor = m(y_true, y_pred)
    return np.float(ssim_tensor)
      
def compute_fsim(img0, img1, nlevels=5, nwavelets=16, L=None):
    """
    https://github.com/tomography/xdesign/blob/master/xdesign/metrics.py
    """

def vifp(ref, dist):
    ref, dist = np.array(ref), np.array(dist)
    sigma_nsq=2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
       
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
                
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0
        
        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0
        
        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps
        
        num += numpy.sum(numpy.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += numpy.sum(numpy.log10(1 + sigma1_sq / sigma_nsq))
        
    result = num/den
    return result

if __name__ == "__main__":
    img_rows, img_cols, batch_size = 224, 224, 32
    dataset = nzLS(split="all")
    data_loader = DataLoader(dataset, batch_size, num_workers=4,
                             shuffle=False)
    batch_iterator = iter(data_loader)
    x, y_true = next(batch_iterator)
    # add one row of noice in the middle
    y_pred = np.copy(y_true)
    y_pred[:, :, img_rows // 2, :] = 1
    y_true = torch.FloatTensor(y_true)
    y_pred = torch.FloatTensor(y_pred)

    matrix = confusion_matrix(y_pred, y_true)
    print('confusion:', matrix)

    hs = hausdorff(y_pred, y_true)
    print('hausdorff:', hs)
