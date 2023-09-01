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


def eval(opt, test_output_dir='test_latest'):
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
    gt_image_dir = result_image_dir
    post_output_dir1 = os.path.join(opt.results_dir, opt.name, 'temp1')
    post_output_dir2 = os.path.join(opt.results_dir, opt.name, 'temp2')
    # if not os.path.isdir(post_output_dir):
    #     os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    if 'pix2pix' in opt.model:
        end_word = 'fake_SB.png'
    else:
        end_word = 'fake_SB.png'
    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png
        if not image_name.endswith(end_word):
            continue
        temp_str = image_name.split('_')
        if temp_str[1] == 'fake':
            continue
        # 初始化操作
        count += 1
        image_num = temp_str[1] + '_' + temp_str[2]
        # gt_image_name = image_num + 'B_reg.jpg'
        gt_image_name = image_num + '.jpg'
        image_path = os.path.join(result_image_dir, image_name)
        gt_image_path = image_path.replace('fake_SB', 'real_SB')
        # gt_image_path = image_path


        # 读取图像
        gt_image = cv2.imread(gt_image_path)
        # h, w, c = gt_image.shape
        # gt_image = gt_image[:, int(w / 2):, :]
        # gt_image = gt_image[:, :int(w / 2), :]
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
        gt_image = gt_image * mask
        # mask_image = images

        # # TODO:不需要的时候可以注释
        # # 保存mask_image
        # # output_image_name = image_num + '_' + image_name.split('_')[2] + '_' + 'B_reg.jpg'
        temp_code = image_name.split('_')[0]
        cv2.imwrite(os.path.join(post_output_dir1,image_name), gt_image)
        cv2.imwrite(os.path.join(post_output_dir2, image_name), mask_image)

        if count == 10 or count % 100 == 0:
            print(count)
        # -------------评价代码-------------
        ssim = structural_similarity(gt_image, mask_image, multichannel=True)
        if not dict_sum_ssim.get(temp_code):
            dict_sum_ssim[temp_code] = 0
        if not dict_sum_max.get(temp_code):
            dict_sum_max[temp_code] = (0, 0)
        if dict_sum_max[temp_code][1] < ssim:
            dict_sum_max[temp_code] = (image_name, ssim)
        # dict_sum_ssim[temp_code] += ssim
        dict_sum_ssim[temp_code] += ssim
        sum_ssim += ssim
        # -------------评价代码-------------
    # print(dict_sum_max)
    # print(dict_sum_ssim)
    print('ssim', sum_ssim / count)
    return sum_ssim / count

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test_total options
    eval(opt)