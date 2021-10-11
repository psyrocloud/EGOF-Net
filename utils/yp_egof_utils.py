# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 22:31:35 2021

@author: psyrocloud
"""

import numpy as np
import cv2 as cv
#import torch


def generate_random_2d_idx_with_mask_robust(mask, samples=4000):
    # filter out repeat elements in udix and vidx
    '''
    If len(valid(mask))<samples, samples = this count
    '''
    # return uidx, vidx for uv to random sample
    h, w = mask.shape
    mm = mask>0
    total_count = np.sum(mask>0)
    if total_count < 10:
        raise ValueError('yp function error: \ngenerate_random_2d_idx_with_mask_robust(): total_count in mask should >= 10')
    flatten_idx = np.random.randint(0, total_count, samples)
    flatten_idx = np.unique(flatten_idx)  # tick repetitive elements
    mask_idx = np.nonzero(mm)
    vidx = mask_idx[0][flatten_idx]
    uidx = mask_idx[1][flatten_idx]
    if len(vidx)<1:
        raise ValueError('yp function error: sampled points are too few (0).')
    return uidx, vidx


def findFundamentalMatrixWithRandomFlow(flol, msk, rt=1.5):
    # inputs are batched tensors, batch must be 1
    # all inputs are torch.tensor, images are between [0, 255]
    is_cuda = flol.is_cuda
    if is_cuda:
        device = flol.device
    _,_,h,w = flol.shape
    flol = flol[0].permute(1,2,0).cpu().numpy()
    msk = msk[0].cpu().numpy()
    vu0, vv0 = np.meshgrid(np.linspace(0, w, w, False), np.linspace(0, h, h, False))
    uv0 = np.stack((vu0,vv0), 2)  # (h, w, 2)
    uv1 = flol + uv0
    # rt = 1.5  # ransac threshold  # sqrt(2)
    # rand sample
    uidx, vidx = generate_random_2d_idx_with_mask_robust(msk, 2000)
    uv0v_r = uv0[vidx, uidx]
    uv1v_r = uv1[vidx, uidx]
    fmat1, robust_mask = cv.findFundamentalMat(uv0v_r, uv1v_r, cv.FM_LMEDS)
    return fmat1