# -*- coding: UTF-8 -*-
"""
@Function:将原始的target和gt相比
@File: evaluation_target.py
@Date: 2021/11/20 16:53 
@Author: Hever
"""
import re
from scipy import ndimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import os
import cv2
import numpy as np

def evaluation(root_dir = './datasets/cataract_0830'):
    # 初始化

    image_size = (512, 512)

    target_gt_dir = os.path.join(root_dir, 'target_gt_post')
    target_dir = os.path.join(root_dir, 'target')
    post_output_dir = os.path.join(root_dir, 'target_post')

    image_name_list = [img for img in os.listdir(target_dir) if img.find('.jpg') > 0 or img.find('.png') > 0]
    print(len(image_name_list))
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_min = {}

    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png

        count += 1
        image_num = re.findall(r'[0-9]+', image_name)[0]
        # gt_image_name = image_num + 'B_reg.jpg'
        gt_image_name = image_num + 'B.png'
        image_path = os.path.join(target_dir, image_name)
        gt_image_path = os.path.join(target_gt_dir, gt_image_name)

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
        temp_code = image_name.split('A')[0]
        cv2.imwrite(os.path.join(post_output_dir, image_name.replace('jpg', 'png')), mask_image)

        if count == 10 or count % 100 == 0:
            print(count)
        # # -------------评价代码-------------
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        if not dict_sum_ssim.get(temp_code):
            dict_sum_ssim[temp_code] = 0
        if not dict_sum_min.get(temp_code):
            dict_sum_min[temp_code] = (1, 20)
        if dict_sum_min[temp_code][1] > ssim:
            dict_sum_min[temp_code] = (psnr, ssim)
        # dict_sum_ssim[temp_code] += ssim
        dict_sum_ssim[temp_code] += ssim
        sum_ssim += ssim
        sum_psnr += psnr
        # -------------评价代码-------------
    print(dict_sum_min)
    print(dict_sum_ssim)
    min_sum_ssim = min_sum_psnr = 0
    for k, (p, s) in dict_sum_min.items():
        min_sum_ssim += s
        min_sum_psnr += p
    print('ssim', min_sum_ssim / len(dict_sum_min))
    print('psnr', min_sum_psnr / len(dict_sum_min))

    # if cataract_test_dataset is not None:
    #     device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    #     is_mean, is_std, fid = get_is_fid_score(cataract_test_dataset, device=device, dataroot=opt.dataroot)
    #     with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #         f.write('%f,%f,%f,' % (is_mean, is_std, fid))

    print('ssim', sum_ssim / count)
    print('psnr', sum_psnr / count)
    return sum_ssim / count


if __name__ == '__main__':
    evaluation()

