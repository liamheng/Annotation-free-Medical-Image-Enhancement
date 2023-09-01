# -*- coding: UTF-8 -*-
"""
@Function:
@File: fundus_seg_dataset.py
@Date: 2021/7/14 13:26 
@Author: Hever
"""
import os.path
import random
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms

def get_params(opt, is_train=True):
    assert opt.crop_size == 256
    if not is_train:
        return {'load_size': opt.crop_size, 'crop_pos': (0, 0), 'flip': False, 'flip_vertical': False}

    new_h = new_w = random.choice([256, 286, 306, 326])

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    flip = random.random() > 0.5
    flip_vertical = random.random() > 0.5
    return {'load_size': new_h, 'crop_pos': (x, y), 'flip': flip, 'flip_vertical': flip_vertical}

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __flip_vertical(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def get_transform_seg_channel(opt, params, method=Image.NEAREST, is_train=True):
    transform_list = []
    mask_transform_list = []
    load_size = params['load_size']
    osize = [load_size, load_size]
    transform_list.append(transforms.Resize(osize, method))
    # TODO:对图像进行阈值处理
    # mask_transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    mask_transform_list.append(transforms.Resize(osize, Image.NEAREST))
    if is_train:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
        mask_transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
        mask_transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

        transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip_vertical'])))
        mask_transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip_vertical'])))
        # TODO：不加入color jitter
        # color_jitter = transforms.ColorJitter(
        #     brightness=0.05,
        #     contrast=0.05,
        #     saturation=0.05,
        #     hue=0.05,
        # )
        # transform_list.append(color_jitter)


    transform_list += [transforms.ToTensor()]
    mask_transform_list += [transforms.ToTensor()]

    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list), transforms.Compose(mask_transform_list)


class FundusSegDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # for eyeq
        self.is_train = opt.isTrain
        if self.is_train:
            self.image_dir = os.path.join(opt.dataroot, 'remove_bad_image')  # get the images directory
            # self.mask_dir = os.path.join(opt.dataroot, 'source_mask')  # get the images directory
            self.ves_dir = os.path.join(opt.dataroot, 'remove_bad_mask')  # get the images directory
        else:
            # self.image_dir = os.path.join(opt.dataroot, 'target_gt')  # get the images directory
            self.image_dir = os.path.join(opt.dataroot, 'target_image_multi_seg')  # get the images directory
            # self.mask_dir = os.path.join(opt.dataroot, 'target_mask')  # get the images directory
            self.ves_dir = os.path.join(opt.dataroot, 'target_mask')  # get the images directory

        self.image_paths = sorted(make_dataset(self.image_dir, opt.max_dataset_size))  # get images paths
        # self.mask_paths = sorted(make_dataset(self.ves_dir, opt.max_dataset_size))  # get images paths
        self.image_size = len(self.image_paths)
        assert(self.opt.load_size == self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        self.is_train = opt.isTrain
        # TODO：直接在这里写预处理

    def __getitem__(self, index):
        # if self.is_train:
        image_path = self.image_paths[index]
        image_name = os.path.split(image_path)[-1].split('-')[0].replace('.png', '') + '.png'
        # mask_path = os.path.join(self.mask_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        # mask = Image.open(mask_path).convert('L')
        #
        transform_params = get_params(self.opt, is_train=self.is_train)
        transform_image, transform_mask = get_transform_seg_channel(self.opt, transform_params, method=Image.BICUBIC,
                                                                    is_train=self.is_train)
        image = transform_image(image)
        # mask = transform_mask(mask)
        if self.is_train:
            ves_path = os.path.join(self.ves_dir, image_name)
            ves = Image.open(ves_path).convert('L')
            ves = transform_mask(ves)
            # th = 0.4
            # ves[ves>=th]=1.0
            # ves[ves<th]=0.0
            # 将脉络膜过滤掉

            ves[ves >= 0.8] = 0
        else:
            ves = None
        # return {'images': images, 'ves': ves, 'mask': mask, 'image_path': image_path}
        return {'images': image, 'ves': ves, 'image_path': image_path}
        #
        # # 测试时必须要有mask
        # if not self.is_train:
        #     image_path = self.image_paths[index]
        #     image_name = os.path.split(image_path)[-1].split('-')[0].replace('.png', '') + '.png'
        #     images = Image.open(image_path).convert('RGB')
        #
        #     mask_path = os.path.join(self.mask_dir, image_name)
        #     mask = Image.open(mask_path).convert('L')
        #
        #     transform_params = get_params(self.opt, is_train=False)
        #     transform_image, transform_mask = get_transform_seg_channel(self.opt, transform_params)
        #     images = transform_image(images)
        #     mask = transform_mask(mask)
        #
        #     return {'images': images, 'mask': mask, 'image_path': image_path}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

