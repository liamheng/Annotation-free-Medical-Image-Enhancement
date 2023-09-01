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

# input_dir_path = 'images'
# output_dir_path = 'concat'
# if not os.path.isdir(output_dir_path):
#     os.mkdir(output_dir_path)
# image_list = os.listdir(input_dir_path)
# fake_path_list = sorted([os.path.join(input_dir_path, name) for name in image_list if name.find('fake_SB') >= 0])
# # real_SA_path_list = sorted([os.path.join(input_dir_path, name) for name in image_list if name.find('real_SA') >= 0])
# real_path_list = sorted([os.path.join(input_dir_path, name) for name in image_list if name.find('real_SB') >= 0])
# for (fake_path, real_path) in zip(fake_path_list, real_path_list):
#     fake = cv2.imread(fake_path)
#     real = cv2.imread(real_path)
#     res = np.hstack([fake, real])
#     cv2.imwrite(fake_path.replace('_fake_SB', '').replace(input_dir_path, output_dir_path), res)
input_dir_path = r'D:\Project\pixDA\pixDA_GM\results\train450_1_1_drive_cataract_low_quality_sim_gan_0720_SimGAN\test_latest\images'
output_dir_path = r'D:\Project\pixDA\pixDA_GM\datasets\drive_cataract_low_quality_luo_0721\source'
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
    image_name_split = os.path.split(fake_path)[-1].replace('_fake_SB.png', '').split('-')
    image_name = image_name_split[0]
    image_num = int(image_name_split[1]) + 16
    image_output_path = os.path.join(output_dir_path, '{}-{}.png'.format(image_name, image_num))
    cv2.imwrite(image_output_path, res)
