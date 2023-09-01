"""
先运行test获取运行结果，此处运行需要输入模型的输出文件夹
在对应的target_gt中取mask（512），mask resize到256，应用在fake TB上，并且保存下来
target_gt再resize到512，之后就能开始计算SSIM
"""

import os
import cv2
import numpy as np
import re
from options.test_options import TestOptions
from scipy import ndimage
from skimage.metrics import structural_similarity

image_size = (256, 256)
index_dict = {'01': ('3A_01_test8_fake_TB.png', 0.7589173118559948),
              '02': ('25A_02_test11_fake_TB.png', 0.7897547000969297),
              '03': ('25A_03_test15_fake_TB.png', 0.7992721958187579),
              '05': ('25A_05_test3_fake_TB.png', 0.7840417249948933),
              '10': ('25A_10_test15_fake_TB.png', 0.7815937669150638),
              '15': ('25A_15_test8_fake_TB.png', 0.7982106512410554),
              '25': ('25A_25_training2_fake_TB.png', 0.788947539827431),
              '27': ('25A_27_training2_fake_TB.png', 0.7923592994287304),
              '29': ('25A_29_training0_fake_TB.png', 0.7979824092698494),
              '34': ('25A_34_training4_fake_TB.png', 0.7964368616506184),
              '35': ('25A_35_training12_fake_TB.png', 0.782954516321368),
              '39': ('25A_39_training9_fake_TB.png', 0.7765491683101867),
              'fake': ('4A_fake_TB.png', 0.8170275796939102),
              'P25': ('25A_P25_112_fake_TB.png', 0.7993405899379645),
              'P32': ('25A_P32_17_fake_TB.png', 0.7807003509412236),
              'S23': ('25A_S23_15_fake_TB.png', 0.7891646937286229),
              'S57': ('25A_S57_16_fake_TB.png', 0.7794720663779316),
              'S61': ('25A_S61_214_fake_TB.png', 0.795846393579616)}
if __name__ == '__main__':
    # 初始化
    opt = TestOptions().parse()  # get test_total options
    result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_latest/images')
    # gt_image_dir = os.path.join(opt.dataroot, 'target_gt')
    gt_image_dir = os.path.join(opt.dataroot, 'target_gt')
    post_output_dir = os.path.join(opt.results_dir, opt.name, 'post_image')
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    if 'cycle' in opt.model:
        end_word = 'fake_B.png'
    else:
        end_word = 'fake_TB.png'
    # for image_name in image_name_list:
    for image_name in index_dict:
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

        # TODO:不需要的时候可以注释
        # 保存mask_image
        # output_image_name = image_num + '_' + image_name.split('_')[2] + '_' + 'B_reg.jpg'
        temp_code = image_name.split('_')[1]
        output_image_name = image_num + '_' + temp_code + '_' + 'B_reg.jpg'
        cv2.imwrite(os.path.join(post_output_dir, 'fake_TB_' + output_image_name), mask_image)

        if count % 10 == 0:
            print(count)
        # -------------评价代码-------------
        ssim = structural_similarity(image, gt_image, data_range=255, multichannel=True)
        if not dict_sum_ssim.get(temp_code):
            dict_sum_ssim[temp_code] = 0
        if not dict_sum_max.get(temp_code):
            dict_sum_max[temp_code] = (0, 0)
        if dict_sum_max[temp_code][1] < ssim:
            dict_sum_max[temp_code] = (image_name, ssim)
        dict_sum_ssim[temp_code] += ssim
        sum_ssim += ssim
        # -------------评价代码-------------
    print(dict_sum_max)
    print(dict_sum_ssim)
    print('ssim', sum_ssim / count)
