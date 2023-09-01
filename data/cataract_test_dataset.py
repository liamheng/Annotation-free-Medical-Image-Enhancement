# -*- coding: UTF-8 -*-
"""
@Function:
@File: cataract_test_dataset.py
@Date: 2021/4/8 19:29 
@Author: Hever
"""
import os.path
import random
# import torch
import re
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import PIL
from PIL import Image
from scipy import ndimage


def mul_mask(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = ndimage.binary_opening(gray > 10, structure=np.ones((8, 8)))

    image = np.transpose(image, (2, 0, 1))
    image = image * mask
    image = np.transpose(image, (1, 2, 0))

    image = Image.fromarray(image)
    return image

class CataractTestDataset(data.Dataset):
    """A dataset class for paired images dataset.

    It assumes that the directory '/path/to/images/train' contains images pairs in the form of {A,B}.
    During test_total time, you need to prepare a directory '/path/to/images/test_total'.
    """

    def __init__(self, opt, test_output_dir='test_latest', mode=1, mul_mask=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.mode = mode  # mode=1只获得一张结果图像, mode=2可获得两张图像
        # 图像保存的目录
        if opt.load_iter != 0:
            self.result_image_dir = os.path.join(opt.results_dir, opt.name,
                                            'test_latest_iter' + str(opt.load_iter) + '/images')
        else:
            if 'result' in test_output_dir:
                self.result_image_dir = os.path.join(test_output_dir, 'images')
            else:
                self.result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')

        if opt.target_gt_dir is not None:
            self.gt_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir)
        else:
            self.gt_image_dir = os.path.join(opt.dataroot, 'target_gt')

        image_name_list = os.listdir(self.result_image_dir)

        if 'pix2pix' in opt.model or 'cycle' in opt.model:
            A_end_word = 'real_A.png'
            end_word = 'fake_B.png'
            # end_word = 'real_A.png'
        else:
            # end_word = 'fake_TB.png'
            A_end_word = 'real_TA.png'  # TA
            end_word = 'fake_TB.png'
        self.real_A_list = []
        self.target_list = []
        self.target_gt_list = []
        for image_name in image_name_list:
            # TODO:对于cycle应该是fake_B.png
            if not image_name.endswith(end_word):
                continue
            self.target_list.append(image_name)
            image_num = re.findall(r'[0-9]+', image_name)[0]
            # gt_image_name = image_num + 'B_reg.jpg'
            gt_image_name = image_num + 'B.jpg'
            self.target_gt_list.append(gt_image_name)
        for image_name in image_name_list:
            if not image_name.endswith(A_end_word):
                continue
            self.real_A_list.append(image_name)
        self.transform_list = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
        self.transform_list_zero_one = transforms.Compose([
            transforms.ToTensor()
        ])
        self.mul_mask = mul_mask
        # ---------------前期准备------------------


    def __getitem__(self, index):
        if self.mode == 1:
            image_path = os.path.join(self.result_image_dir, self.target_list[index])
            # gt_image_path = os.path.join(self.gt_image_dir, self.target_gt_list[index])
            image_B = Image.open(image_path).convert('RGB')
            if self.mul_mask:
                image_B = mul_mask(image_B)
                # image_B.save('./images/' + self.target_list[index])
            image_B = self.transform_list(image_B)
            return image_B
        elif self.mode == 2:
            image_path = os.path.join(self.result_image_dir, self.target_list[index])
            image_B = Image.open(image_path).convert('RGB')
            if self.mul_mask:
                image_B = mul_mask(image_B)
            image_B = self.transform_list(image_B)
            image_A_path = os.path.join(self.result_image_dir, self.real_A_list[index])
            image_A = Image.open(image_A_path).convert('RGB')
            if self.mul_mask:
                image_A = mul_mask(image_A)
            image_A = self.transform_list(image_A)
            return image_A, image_B
        elif self.mode == 3:
            image_path = os.path.join(self.result_image_dir, self.target_list[index])
            image_A_path = os.path.join(self.result_image_dir, self.real_A_list[index])
            return image_A_path, image_path
        elif self.mode == 4:
            image_path = os.path.join(self.result_image_dir, self.target_list[index])
            image_B = Image.open(image_path).convert('RGB')
            if self.mul_mask:
                image_B = mul_mask(image_B)
            image_B = self.transform_list_zero_one(image_B)
            return image_B
        elif self.mode == 5:
            image_path = os.path.join(self.result_image_dir, self.target_list[index])
            return image_path


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.target_list)
