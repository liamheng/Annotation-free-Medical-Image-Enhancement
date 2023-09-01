# -*- coding: UTF-8 -*-
"""
@Function:由于不能使用与文件夹名称同名的py文件名，因此使用这个py文件来运行
@File: evaluation.py
@Date: 2021/4/12 14:41 
@Author: Hever
"""


import os
import cv2
import numpy as np
import re
import torch
from options.test_options import TestOptions
from model_eval.fid_score import get_is_fid_score
from model_eval.eval_public import get_is_fid, get_brisque, get_niqe
from scipy import ndimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def eval(opt, test_output_dir='test_latest', cataract_test_dataset=None):
    # 初始化
    image_size = (256, 256)

    if opt.load_iter != 0:
        result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_latest_iter' + str(opt.load_iter) + '/images')
    else:
        # in是为了训练时测试，not in是为了直接test时使用的
        if 'result' in test_output_dir:
            result_image_dir = os.path.join(test_output_dir, 'images')
        else:
            result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')
    # gt_image_dir = os.path.join(opt.dataroot, 'target_gt')
    if opt.target_gt_dir is not None:
        gt_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir)
    else:
        gt_image_dir = os.path.join(opt.dataroot, 'target_gt')
    post_output_dir = os.path.join(opt.results_dir, opt.name, 'post_image')
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    if 'pix2pix' in opt.model or 'cycle' in opt.model:
        end_word = 'fake_B.png'
    else:
        end_word = 'fake_TB.png'
    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png
        if not image_name.endswith(end_word):
            continue
        # 初始化操作
        count += 1
        image_num = re.findall(r'[0-9]+', image_name)[0]
        # gt_image_name = image_num + 'B_reg.jpg'
        gt_image_name = image_num + 'B.jpg'
        image_path = os.path.join(result_image_dir, image_name)
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)

        # 读取图像
        gt_image = cv2.imread(gt_image_path)
        image = cv2.imread(image_path)

        # 取mask
        image_gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)
        gray = np.array(image_gray)
        threshold = 5
        if '10' in image_name:
            threshold = 10
        mask = ndimage.binary_opening(gray>threshold, structure=np.ones((8,8)))

        # resize gt和mask到256，并应用到image中
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, image_size)
        mask = mask[:, :, np.newaxis]
        gt_image = cv2.resize(gt_image, image_size)
        mask_image = image * mask

        # # TODO:不需要的时候可以注释
        # # 保存mask_image
        # # output_image_name = image_num + '_' + image_name.split('_')[2] + '_' + 'B_reg.jpg'
        temp_code = image_name.split('_')[0]
        cv2.imwrite(os.path.join(post_output_dir, image_name), mask_image)

        if count == 10 or count % 100 == 0:
            print(count)
        # -------------评价代码-------------
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        # if not dict_sum_ssim.get(temp_code):
        #     dict_sum_ssim[temp_code] = 0
        # if not dict_sum_max.get(temp_code):
        #     dict_sum_max[temp_code] = (0, 0)
        # if dict_sum_max[temp_code][1] < ssim:
        #     dict_sum_max[temp_code] = (image_name, ssim)
        # # dict_sum_ssim[temp_code] += ssim
        # dict_sum_ssim[temp_code] += ssim
        sum_ssim += ssim
        sum_psnr += psnr
        # -------------评价代码-------------
    # print(dict_sum_max)
    # print(dict_sum_ssim)


    if cataract_test_dataset is not None:
        get_is_fid(opt, cataract_test_dataset)
        get_brisque(opt, cataract_test_dataset)
        get_niqe(opt, cataract_test_dataset)
        # device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
        # is_mean, is_std, fid = get_is_fid_score(cataract_test_dataset, device=device)
        #
        # with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        #     f.write('%f,%f,%f,' % (is_mean, is_std, fid))
    with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        f.write('%f,%f\n' % (sum_ssim / count, sum_psnr / count))
    print('ssim', sum_ssim / count)
    print('psnr', sum_psnr / count)
    return sum_ssim / count

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test_total options
    eval(opt)