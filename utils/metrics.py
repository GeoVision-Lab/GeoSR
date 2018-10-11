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
from datasets import *
from torch.utils.data import DataLoader

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
    return 10 * torch.log10(1 - mse(y_pred, y_true, threshold=None))


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
