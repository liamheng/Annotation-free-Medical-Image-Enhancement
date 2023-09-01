# -*- coding: UTF-8 -*-
"""
@Function:
@File: seg_loader.py
@Date: 2022/1/14 9:52 
@Author: Hever
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageCms
import os
import torchvision.transforms as transforms

# from sklearn import preprocessing
# import pandas as pd

class SegCataractInferenceDataset(Dataset):
    def __init__(self, data_dir, mask_dir, gt_dir='', gt_mask_dir='', is_fake_B=False, is_fake_TB=False):
        self.data_dir = data_dir  # 是图像数据的dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.gt_mask_dir = gt_mask_dir
        self.image_names = []
        self.image_names = os.listdir(data_dir)
        if is_fake_B:
            self.image_names = [image_name for image_name in self.image_names
                                if image_name.find('fake_B.') >= 0]
        if is_fake_TB:
            self.image_names = [image_name for image_name in self.image_names
                                if image_name.find('fake_TB.') >= 0]
        # self.image_paths = [os.path.join(data_dir, image_name) for image_name in image_names]
        self.transform_image = transforms.Compose(
            [transforms.Resize((256, 256), Image.BICUBIC), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.transform_mask = transforms.Compose(
            [transforms.Resize((256, 256), Image.BICUBIC), transforms.ToTensor()]
        )


    def __getitem__(self, index):
        image_name = self.image_names[index]
        mask_name = image_name.split('.')[0].split('-')[0].split('_')[0] + '.png'
        image_path = os.path.join(self.data_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        th = 0.4
        mask[mask >= th] = 1.0
        mask[mask < th] = 0.0
        if self.gt_dir != '':
            gt_mask_name = image_name.split('.')[0].split('-')[0].split('_')[0].replace('A', 'B') + '.png'
            gt_path = os.path.join(self.gt_dir, gt_mask_name)
            gt_mask_path = os.path.join(self.gt_mask_dir, gt_mask_name)
            gt = Image.open(gt_path).convert('L')
            gt_mask = Image.open(gt_mask_path).convert('L')
            gt = self.transform_mask(gt)
            gt_mask = self.transform_mask(gt_mask)
            th = 0.4
            gt[gt >= th] = 1.0
            gt[gt < th] = 0.0
            gt_mask[gt_mask >= th] = 1.0
            gt_mask[gt_mask < th] = 0.0
            mask_union = mask * gt_mask
            return {
                'images': image, 'mask_union': mask_union, 'gt': gt, 'image_name': image_name
            }

        return {
            'images': image, 'mask': mask, 'image_name': image_name
        }

    def __len__(self):
        return len(self.image_names)