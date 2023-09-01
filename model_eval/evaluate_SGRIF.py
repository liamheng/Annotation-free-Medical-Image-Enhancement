"""
先运行test获取运行结果，此处运行需要输入模型的输出文件夹
在对应的target_gt中取mask（512），mask resize到256，应用在fake TB上，并且保存下来
target_gt再resize到512，之后就能开始计算SSIM
"""

import os
import cv2
import numpy as np
import re
import torch
from options.test_options import TestOptions
# from model_eval.fid_score import get_is_fid_score
from scipy import ndimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
# from model_eval.eval_public import eval_public
# from comparison.cataract_comparison_test_dataset import CataractComparisonTestDataset

def model_eval(opt, test_output_dir='test_latest', meters=None, wrap=True, write_res=True):
    # 初始化
    image_size = (256, 256)
    # image_size = (opt.crop_size, opt.crop_size)
    print('evaluating in the images size of {}'.format(image_size))
    # if opt.load_iter != 0:
    #     result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_latest_iter' + str(opt.load_iter) + '/images')
    # else:
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
    if 'pix2pix' in opt.model or 'cycle' in opt.model or 'HFC2stage' == opt.model:
        end_word = 'fake_TB.png'
    elif 'SGRIF' in opt.name:
        end_word = '.jpg'
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
        if 'DRIVE_simulated' in opt.dataroot:
            gt_image_name = image_num + 'B.png'
        elif 'fiq' in opt.dataroot:
            gt_image_name = image_num + '.png'
        else:
            gt_image_name = image_num + 'B.jpg'

        # gt_image_name = image_num + '.png'

        image_path = os.path.join(result_image_dir, image_name)
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)

        # 读取图像
        gt_image = cv2.imread(gt_image_path)
        if gt_image is None:
            gt_image = cv2.imread(gt_image_path.replace('jpg', 'png'))
        image = cv2.imread(image_path)
        image = cv2.resize(image, (opt.crop_size, opt.crop_size))

        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        image = cv2.resize(image, image_size)

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
        # # 分块进行评价
        # if opt.crop_size > 256:
        #     ssim = psnr = 0
        #     for i in range(int(opt.crop_size / 256)):
        #         for j in range(int(opt.crop_size / 256)):
        #             part_of_mask_image = mask_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             part_of_gt_image = gt_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             ssim += structural_similarity(part_of_mask_image, part_of_gt_image, data_range=255, multichannel=True)
        #             psnr += peak_signal_noise_ratio(part_of_gt_image, part_of_mask_image, data_range=255)
        #     ssim /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        #     psnr /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        # else:
        #     # -------------评价代码-------------
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        if not dict_sum_ssim.get(temp_code):
            dict_sum_ssim[temp_code] = 0
        if not dict_sum_max.get(temp_code):
            dict_sum_max[temp_code] = (0, 0)
        if dict_sum_max[temp_code][1] < ssim:
            dict_sum_max[temp_code] = (image_name, ssim)
        # dict_sum_ssim[temp_code] += ssim
        dict_sum_ssim[temp_code] += ssim
        sum_ssim += ssim
        sum_psnr += psnr
        # -------------评价代码-------------
    # print(dict_sum_max)
    # print(dict_sum_ssim)


    # if cataract_test_dataset is not None:
    #     device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    #     is_mean, is_std, fid = get_is_fid_score(cataract_test_dataset, device=device, dataroot=opt.dataroot)
    #     with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #         f.write('%f,%f,%f,' % (is_mean, is_std, fid))
    if write_res:
        # with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        with open(os.path.join('./results', 'log', opt.name + '.csv'), 'a') as f:
            if not wrap:
                f.write('%f,%f,' % (sum_ssim / count, sum_psnr / count))
                if meters is not None:
                    for name, meter in meters.meters.items():
                        f.write('%.4f,' % meter.global_avg)
            else:
                f.write('%f,%f,' % (sum_ssim / count, sum_psnr / count))
                if meters is not None:
                    for name, meter in meters.meters.items():
                        f.write('%.4f,' % meter.global_avg)
                f.write('\n')
    print('Number for process ssim and psnr:{}'.format(count))
    print('ssim', sum_ssim / count)
    print('psnr', sum_psnr / count)
    return sum_ssim / count, sum_psnr / count

def fiq_evaluation(opt, test_output_dir='test_latest'):
    # 初始化
    image_size = (256, 256)
    # image_size = (opt.crop_size, opt.crop_size)
    print('evaluating in the images size of {}'.format(image_size))
    # test_output_dir = 'test_latest'
    if opt.load_iter != 0:
        result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_fiq_latest_iter' + str(opt.load_iter) + '/images')
    else:
        # in是为了训练时测试
        if 'result' in test_output_dir:
            result_image_dir = os.path.join(test_output_dir, 'images')
        else:
            result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')
    if opt.target_gt_dir is not None:
        gt_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir)
        gt_mask_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir.replace('_mask', '') + '_mask')
    else:
        gt_image_dir = os.path.join(opt.dataroot, 'target_gt')
        gt_mask_image_dir = os.path.join(opt.dataroot, 'target_gt_mask')

    # 可视化
    post_output_dir = os.path.join(opt.results_dir, opt.name, 'post_image')
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    # 前期准备
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    # end_word = 'fake_B.png'
    if 'pix2pix' in opt.model or 'cycle' in opt.model:
        end_word = 'fake_TB.png'
    elif 'SGRIF' in opt.name:
        end_word = '.png'
    else:
        end_word = 'fake_TB.png'
    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png
        if not image_name.endswith(end_word):
            continue
        # 初始化操作
        image_num = re.findall(r'[0-9]+', image_name)[0]
        gt_image_name = image_num + '.png'
        if 'avr_test' in opt.target_gt_dir:
            gt_image_name = image_name.split('-')[0] + '.png'

        image_path = os.path.join(result_image_dir, image_name)
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)
        mask_path = os.path.join(gt_mask_image_dir, gt_image_name)
        # 读取图像
        try:
            gt_image = cv2.imread(gt_image_path)
            if gt_image is None:
                raise Exception('no gt images')
        except:
            continue
        count += 1

        gt_image = cv2.resize(gt_image, image_size)
        image = cv2.imread(image_path)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        image = cv2.resize(image, (opt.crop_size, opt.crop_size))
        image = cv2.resize(image, image_size)

        # 读取mask并进行预处理
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
        mask = mask / 255
        mask = mask.astype(np.uint8)
        mask = mask[:, :, np.newaxis]

        mask_image = image * mask
        gt_image = gt_image * mask

        cv2.imwrite(os.path.join(post_output_dir, image_name), mask_image)

        if count == 10 or count % 100 == 0:
            print(count)

        # 评价
        # if opt.crop_size > 256:
        #     ssim = psnr = 0
        #     for i in range(int(opt.crop_size / 256)):
        #         for j in range(int(opt.crop_size / 256)):
        #             part_of_mask_image = mask_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             part_of_gt_image = gt_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             ssim += structural_similarity(part_of_mask_image, part_of_gt_image, data_range=255, multichannel=True)
        #             psnr += peak_signal_noise_ratio(part_of_gt_image, part_of_mask_image, data_range=255)
        #     ssim /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        #     psnr /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        # else:
        #     # -------------评价代码-------------
        #     ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        #     psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)

        sum_ssim += ssim
        sum_psnr += psnr
    print('Number for process ssim and psnr:{}'.format(count))

    print('Test result: ssim: {:.3f}, psnr: {:.2f}'.format(sum_ssim / count, sum_psnr / count))
    return sum_ssim / count, sum_psnr / count

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test_total options
    model_eval(opt)
    # cataractTestDataset = CataractTestDataset(opt, test_web_dir)
    # eval_public()