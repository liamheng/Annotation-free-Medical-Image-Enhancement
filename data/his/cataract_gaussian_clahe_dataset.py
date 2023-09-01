import os.path
import random
import torch
import numpy as np
import cv2
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_four_channel
from data.image_folder import make_dataset
from PIL import Image


class CataractGaussianCLAHEDataset(BaseDataset):
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
        self.dir_source = os.path.join(opt.dataroot, 'source')  # get the images directory
        self.dir_target = os.path.join(opt.dataroot, 'target')  # get the images directory

        self.source_paths = sorted(make_dataset(self.dir_source, opt.max_dataset_size))  # get images paths
        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get images paths

        self.target_size = len(self.target_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain
        # self.clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
        # print('creating clahe')

    def __getitem__(self, index):
        """Return a images point and its metadata information.

        Parameters:
            index - - a random integer for images indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an images in the input domain
            B (tensor) - - its corresponding images in the target domain
            A_paths (str) - - images paths
            B_paths (str) - - images paths (same as A_paths)
        """
        # read a images given a random integer index
        source_path = self.source_paths[index]
        target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]

        SAB = Image.open(source_path).convert('RGB')
        TA = Image.open(target_path).convert('RGB')
        w, h = SAB.size
        w2 = int(w / 2)
        SA = SAB.crop((0, 0, w2, h))
        SB = SAB.crop((w2, 0, w, h))
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
        SA_g_channel = np.array(SA)[:, :, 1]
        SA_g_channel = clahe.apply(SA_g_channel)
        SA_g_channel = Image.fromarray(SA_g_channel)
        TA_g_channel = np.array(TA)[:, :, 1]
        TA_g_channel = clahe.apply(TA_g_channel)
        TA_g_channel = Image.fromarray(TA_g_channel)


        # 对输入和输出进行同样的transform（裁剪也继续采用）
        source_transform_params = get_params(self.opt, SA.size)
        source_A_transform, source_A_clahe_transform = get_transform_four_channel(self.opt, source_transform_params, grayscale=(self.input_nc == 1))
        source_B_transform = get_transform(self.opt, source_transform_params, grayscale=(self.output_nc == 1))

        target_transform_params = get_params(self.opt, TA.size, is_source=False)
        # target_A_transform = get_transform(self.opt, target_transform_params, grayscale=(self.input_nc == 1))
        target_A_transform, target_A_clahe_transform = get_transform_four_channel(self.opt, target_transform_params, grayscale=(self.input_nc == 1))

        SA = source_A_transform(SA)
        SA_C = source_A_clahe_transform(SA_g_channel)
        # 使用同一个transform
        SB = source_B_transform(SB)

        TA = target_A_transform(TA)
        TA_C = target_A_clahe_transform(TA_g_channel)


        return {'SA': SA, 'SB': SB, 'SA_path': source_path, 'SA_C': SA_C, 'TA_C': TA_C,
                'SB_path': source_path, 'TA': TA, 'TA_path': target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.source_paths)
