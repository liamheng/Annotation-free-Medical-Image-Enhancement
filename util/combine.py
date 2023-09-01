# -*- coding: UTF-8 -*-
"""
@Function:
@File: combine.py
@Date: 2021/7/11 19:05 
@Author: Hever
"""
import os
import shutil
import cv2
import numpy as np

input_dir_path = 'images'
output_dir_path = 'concat'
if not os.path.isdir(output_dir_path):
    os.mkdir(output_dir_path)
image_list = os.listdir(input_dir_path)
fake_path_list = sorted([os.path.join(input_dir_path, name) for name in image_list if name.find('fake_SB') >= 0])
# real_SA_path_list = sorted([os.path.join(input_dir_path, name) for name in image_list if name.find('real_SA') >= 0])
real_path_list = sorted([os.path.join(input_dir_path, name) for name in image_list if name.find('real_SB') >= 0])
for (fake_path, real_path) in zip(fake_path_list, real_path_list):
    fake = cv2.imread(fake_path)
    real = cv2.imread(real_path)
    res = np.hstack([fake, real])
    cv2.imwrite(fake_path.replace('_fake_SB', '').replace(input_dir_path, output_dir_path), res)

