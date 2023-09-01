import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image


class IseeFullTestDataset(BaseDataset):
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
        self.dir_target = os.path.join('/images/liuhaofeng/Project/classifier/dataset/isee_preprocessed/images/')  # get the images directory
        self.dir_target_mask = os.path.join(opt.dataroot, 'target_mask')  #

        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get images paths
        self.target_mask_paths = sorted(make_dataset(self.dir_target_mask, opt.max_dataset_size))  # get images paths

        self.target_size = len(self.target_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain

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
        target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]
        target_mask_path = '/images/liuhaofeng/Project/pixDA_CA/datasets/cataract_isee_0722/target_mask/99514.png'

        TA = Image.open(target_path).convert('RGB')
        TA_mask = Image.open(target_mask_path).convert('RGB')


        # 对输入和输出进行同样的transform（裁剪也继续采用）
        target_transform_params = get_params(self.opt, TA.size, is_source=False)
        target_A_transform, target_A_mask_transform = get_transform_six_channel(self.opt, target_transform_params, grayscale=(self.input_nc == 1))

        TA = target_A_transform(TA)
        T_mask = target_A_mask_transform(TA_mask)

        return {'SA': TA, 'SB': TA, 'S_mask': T_mask, 'SA_path': target_path,
                'SB_path': target_path, 'TA': TA, 'T_mask': T_mask, 'TA_path': target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.target_paths)

