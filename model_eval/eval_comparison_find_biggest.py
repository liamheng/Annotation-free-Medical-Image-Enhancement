# -*- coding: UTF-8 -*-
"""
@Function:
@File: eval_comparison.py
@Date: 2021/4/30 20:41
@Author: Hever
"""

import os
import cv2
import numpy as np
import re
import torch
import argparse
import shutil
from options.test_options import TestOptions
from model_eval.fid_score import get_is_fid_score
from scipy import ndimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from model_eval.eval_public import eval_public
from comparison.cataract_comparison_test_dataset import CataractComparisonTestDataset


def eval(dataroot='./datasets/cataract_0830', results_dir='./results/clahe_3', full=True):
    # 初始化
    image_size = (256, 256)
    result_image_dir = os.path.join(results_dir)

    gt_image_dir = os.path.join(dataroot, 'target_gt')
    post_output_dir = os.path.join(results_dir, 'post_image')
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = [img for img in os.listdir(result_image_dir) if img.find('jpg') > 0 or img.find('png') > 0]
    if not full:
        image_name_list = [img for img in image_name_list if img.find('_') < 0]
    if len(image_name_list) == 0:
        image_name_list = [img for img in os.listdir(result_image_dir) if img.find('jpg') > 0 or img.find('png') > 0]
    if len(image_name_list) > 500:
        image_name_list = [img for img in image_name_list if img.find('fake_TB.') > 0]
    if results_dir.find('pix2pix') > 0:
        image_name_list = [img for img in image_name_list if img.find('fake_B.') > 0]

    print(len(image_name_list))
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_ssim_min = {}
    dict_ssim_max = {}
    dict_psnr_ssim = {}
    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png

        count += 1
        image_num = re.findall(r'[0-9]+', image_name)[0]
        # gt_image_name = image_num + 'B_reg.jpg'
        gt_image_name = image_num + 'B.jpg'
        image_path = os.path.join(result_image_dir, image_name)
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)

        # 读取图像
        gt_image = cv2.imread(gt_image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        # 取mask
        image_gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)
        gray = np.array(image_gray)
        threshold = 5
        if '10' in image_name:
            threshold = 10
        mask = ndimage.binary_opening(gray > threshold, structure=np.ones((8, 8)))

        # resize gt和mask到256，并应用到image中
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, image_size)
        mask = mask[:, :, np.newaxis]
        gt_image = cv2.resize(gt_image, image_size)
        mask_image = image * mask

        # # TODO:不需要的时候可以注释
        # # 保存mask_image
        # # output_image_name = image_num + '_' + image_name.split('_')[2] + '_' + 'B_reg.jpg'
        temp_code = image_name.split('_')[0].replace('.png', '').replace('.jpg', '')
        cv2.imwrite(os.path.join(post_output_dir, image_name), mask_image)

        if count == 10 or count % 100 == 0:
            print(count)
        # -------------评价代码-------------
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        # if not dict_sum_ssim.get(temp_code):
        #     dict_sum_ssim[temp_code] = 0
        if not dict_ssim_max.get(temp_code):
            dict_ssim_max[temp_code] = (0, 0)
            dict_ssim_min[temp_code] = (1.0, 1.0)
        if dict_ssim_max[temp_code][1] < ssim:
            dict_ssim_max[temp_code] = (image_name, ssim)
        if dict_ssim_min[temp_code][1] > ssim:
            dict_ssim_min[temp_code] = (image_name, ssim)

        if not dict_psnr_ssim.get(temp_code):
            dict_psnr_ssim[temp_code] = (0, 0)
        if dict_psnr_ssim[temp_code][1] < psnr:
            dict_psnr_ssim[temp_code] = (image_name, psnr)

        # dict_sum_ssim[temp_code] += ssim
        # dict_sum_ssim[temp_code] += ssim
        sum_ssim += ssim
        sum_psnr += psnr
        # -------------评价代码-------------
    print(dict_ssim_max)
    print(dict_ssim_min)
    print(dict_psnr_ssim)

    # print(dict_sum_ssim)

    # if cataract_test_dataset is not None:
    #     device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    #     is_mean, is_std, fid = get_is_fid_score(cataract_test_dataset, device=device, dataroot=opt.dataroot)
    #     with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #         f.write('%f,%f,%f,' % (is_mean, is_std, fid))

    print('ssim', sum_ssim / count)
    print('psnr', sum_psnr / count)
    return dict_ssim_max, dict_psnr_ssim


def get_diff(dict1, dict2):
    diff = {}
    for k in dict1.keys():
        d = dict1[k][1] - dict2[k][1]
        diff[k] = (dict1[k][0], d)
    print(diff)
    return diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser('create images pairs')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser.add_argument('--dataroot', type=str, default='./datasets/public_0411')
    parser.add_argument('--name', type=str, default='train240-3-5_pixDA_one_guassian_0830_gaussian_l1g_50_G_L1G_0.05_DD_1_decay_100_RMS_private')
    # parser.add_argument('--name', type=str, default='clahe_private_256')
    # parser.add_argument('--results_dir', type=str, default='./results/dark_channel_public')
    # parser.add_argument('--dataroot', type=str, default='./datasets/cataract_0830')
    # parser.add_argument('--dataroot', type=str, default='./datasets/public_0411')
    parser.add_argument('--dataroot', type=str, default='./datasets/cataract_0830')
    # parser.add_argument('--name', type=str, default='RIE_no_gray_public')
    # parser.add_argument('--results_dir', type=str, default='./results/RIE_no_gray_public')
    # parser.add_argument('--name', type=str, default='RIE05_private_0504')
    parser.add_argument('--results_dir', type=str, default='./results/train240-3-5_pixDA_one_guassian_0830_gaussian_l1g_50_G_L1G_0.05_DD_1_decay_100_RMS_private')
    # parser.add_argument('--results_dir', type=str, default='./results/clahe_private_256')

    parser.add_argument('--full', type=bool, default=True)
    opt = parser.parse_args()  # get test_total options

    RESULT_BETTER_DIR = './results/BETTER_0517'
    if not os.path.isdir(RESULT_BETTER_DIR):
        os.mkdir(RESULT_BETTER_DIR)
    ours_name = 'train240-3-5_pixDA_one_guassian_0830_gaussian_l1g_50_G_L1G_0.05_DD_1_decay_100_RMS_private'
    ours_dir = 'D:\\research\\cataract\\public1+private\\train240-10_pixDA_guassian_no_GDD_0830_l1g_50_DD_1_decay_100_RMS\\test_latest_iter100\\images'
    clahe = 'clahe_private_256'
    clahe_dir = 'D:\\research\\cataract\\public1+private\\train240-8_pixDA_sobel_0830\\test_latest_iter100\\images'

    opt.name = ours_name
    opt.results_dir = ours_dir

    print('evaluating', opt.results_dir)
    dataset = CataractComparisonTestDataset(dataroot=opt.dataroot, results_dir=opt.results_dir)
    # private数据集
    if not os.path.isdir(os.path.join(opt.results_dir, 'log')):
        os.mkdir(os.path.join(opt.results_dir, 'log'))
    if opt.dataroot.find('public') < 0:
        ssim_dict1, psnr_dict1 = eval(dataroot=opt.dataroot, results_dir=opt.results_dir, full=True)

    opt.name = clahe
    opt.results_dir = clahe_dir
    ssim_dict2, psnr_dict2 = eval(dataroot=opt.dataroot, results_dir=opt.results_dir, full=True)
    diff1 = get_diff(ssim_dict1, ssim_dict2)
    # diff2 = get_diff(psnr_dict1, psnr_dict2)
    # for k, v in diff1.items():
    #     shutil.copy(os.path.join(ours_dir, v[0]), os.path.join(RESULT_BETTER_DIR, v[0]))
    # public数据集
    # eval_public(opt, dataset)
