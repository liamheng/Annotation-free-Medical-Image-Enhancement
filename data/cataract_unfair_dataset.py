# -*- coding: UTF-8 -*-
"""
@Function:
@File: cataract_unfair.py
@Date: 2021/4/22 14:46 
@Author: Hever
"""
import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class CataractUnfairDataset(BaseDataset):
    """A dataset class for paired images dataset.

    It assumes that the directory '/path/to/images/train' contains images pairs in the form of {A,B}.
    During test_total time, you need to prepare a directory '/path/to/images/test_total'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_source_A = os.path.join(opt.dataroot, 'source_A')  # get the images directory
        self.dir_source_B = os.path.join(opt.dataroot, 'source_B')  # get the images directory
        self.dir_target = os.path.join(opt.dataroot, 'target')  # get the images directory

        self.source_A_paths = sorted(make_dataset(self.dir_source_A, opt.max_dataset_size))  # get images paths
        self.source_B_paths = sorted(make_dataset(self.dir_source_B, opt.max_dataset_size))  # get images paths
        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get images paths

        self.source_A_size = len(self.source_A_paths)
        self.source_B_size = len(self.source_B_paths)
        self.target_size = len(self.target_paths)

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain


    def __getitem__(self, index):
        # 在unfair数据集中，A的数据（白内障）比B的数据少
        source_A_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.source_A_size
        source_A_path = self.source_A_paths[source_A_index]
        source_B_index = index % self.source_B_size
        source_B_path = self.source_B_paths[source_B_index]
        SA = Image.open(source_A_path).convert('RGB')
        SB = Image.open(source_B_path).convert('RGB')


        # apply the same transform to both A and B
        # 对输入和输出进行同样的transform（裁剪也继续采用）
        source_A_transform_params = get_params(self.opt, SA.size)
        source_A_transform = get_transform(self.opt, source_A_transform_params, grayscale=(self.input_nc == 1))
        # TODO:可以给target做一个独立的transform
        source_B_transform_params = get_params(self.opt, SB.size)
        source_B_transform = get_transform(self.opt, source_B_transform_params, grayscale=(self.input_nc == 1))

        SA = source_A_transform(SA)
        SB = source_B_transform(SB)

        # if not self.isTrain:
        target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]
        TA = Image.open(target_path).convert('RGB')
        target_transform_params = get_params(self.opt, TA.size, is_source=False)
        target_A_transform = get_transform(self.opt, target_transform_params, grayscale=(self.input_nc == 1))
        TA = target_A_transform(TA)
        return {'SA': SA, 'SB': SB, 'SA_path': source_A_path, 'SB_path': source_B_path, 'TA': TA, 'TA_path': target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.isTrain:
            return len(self.source_B_paths) * 6
        else:
            return len(self.target_paths)
