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

def sam(y_pred, y_true, threshold=None):
    """
    Spectral Angle Mapper
    Calculates the angle in spectral space between pixels and a set of reference spectra (endmembers) 
        for image classification based on spectral similarity. 
    """

def compare_nrmse(im_true, im_test, norm_type='Euclidean'):
    """Compute the normalized root mean-squared error (NRMSE) between two
    images.
    Parameters
    ----------
    im_true : ndarray
        Ground-truth image.
    im_test : ndarray
        Test image.
    norm_type : {'Euclidean', 'min-max', 'mean'}
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:
        - 'Euclidean' : normalize by the averaged Euclidean norm of
          ``im_true``::
              NRMSE = RMSE * sqrt(N) / || im_true ||
          where || . || denotes the Frobenius norm and ``N = im_true.size``.
          This result is equivalent to::
              NRMSE = || im_true - im_test || / || im_true ||.
        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``
    Returns
    -------
    nrmse : float
        The NRMSE metric.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    _assert_compatible(im_true, im_test)
    im_true, im_test = _as_floats(im_true, im_test)

    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt(np.mean((im_true*im_true), dtype=np.float64))
    elif norm_type == 'min-max':
        denom = im_true.max() - im_true.min()
    elif norm_type == 'mean':
        denom = im_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return np.sqrt(compare_mse(im_true, im_test)) / denom

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
