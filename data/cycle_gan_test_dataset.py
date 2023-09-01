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


class CycleGANTestDataset(BaseDataset):
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
        self.dir_target = os.path.join(opt.dataroot, 'target')  # get the images directory

        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get images paths

        self.target_size = len(self.target_paths)

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain


    def __getitem__(self, index):
        # if not self.isTrain:
        target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]
        TA = Image.open(target_path).convert('RGB')
        target_transform_params = get_params(self.opt, TA.size, is_source=False)
        target_A_transform = get_transform(self.opt, target_transform_params, grayscale=(self.input_nc == 1))
        TA = target_A_transform(TA)
        return {'SA': TA, 'SB': TA, 'SA_path': target_path, 'SB_path': target_path, 'TA': TA, 'TA_path': target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.target_paths)
