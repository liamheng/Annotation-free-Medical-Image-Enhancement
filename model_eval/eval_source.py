# -*- coding: UTF-8 -*-
"""
@Function:
@File: eval_source.py
@Date: 2021/11/20 20:10 
@Author: Hever
"""

import os
import cv2
import numpy as np
import re
import torch
import argparse
from options.test_options import TestOptions
from model_eval.fid_score import get_is_fid_score
from scipy import ndimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from model_eval.eval_public import eval_public
from comparison.cataract_comparison_test_dataset import CataractComparisonTestDataset

image_dir = './datasets/drive_DG_temp_0911/source'
image_size = (256, 256)
image_name_list = [img for img in os.listdir(image_dir)]
sum_psnr = sum_ssim = count = 0

for image_path in image_name_list:
    path = os.path.join(image_dir, image_path)
    image = cv2.imread(path)
    SA = image[:, :512, :]
    SB = image[:, 512:, :]
    SA = cv2.resize(SA, image_size)
    SB = cv2.resize(SB, image_size)
    ssim = structural_similarity(SA, SB, data_range=255, multichannel=True)
    psnr = peak_signal_noise_ratio(SB, SA, data_range=255)
    sum_ssim += ssim
    sum_psnr += psnr
    count += 1
print('ssim', sum_ssim / count)
print('psnr', sum_psnr / count)