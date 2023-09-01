import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class CataractGaussianDataset(BaseDataset):
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
        self.dir_source_gaussian = os.path.join(opt.dataroot, 'source_gaussian')  # get the images directory
        self.dir_target_gaussian = os.path.join(opt.dataroot, 'target_gaussian')  # get the images directory

        path_source, path_source_gaussian = make_dataset(self.dir_source, opt.max_dataset_size, extra_dir=self.dir_source_gaussian)
        self.source_paths = sorted(path_source)  # get images paths
        self.source_gaussian_paths = sorted(path_source_gaussian)  # get images paths

        path_target, path_target_gaussian = make_dataset(self.dir_target, opt.max_dataset_size,
                                                             extra_dir=self.dir_target_gaussian)  # get images paths
        self.target_paths = sorted(path_target)  # get images paths
        self.target_gaussian_paths = sorted(path_target_gaussian)  # get images paths
        self.target_size = len(self.target_paths)

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
        # assert(self.opt.load_target_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
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
        source_path = self.source_paths[index]
        source_gaussian_path = self.source_gaussian_paths[index]
        target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]
        target_gaussian_path = self.target_gaussian_paths[target_index]

        SAB = Image.open(source_path).convert('RGB')
        TA = Image.open(target_path).convert('RGB')
        SA_G = Image.open(source_gaussian_path).convert('RGB')
        TA_G = Image.open(target_gaussian_path).convert('RGB')

        # split AB images into A and B
        w, h = SAB.size
        w2 = int(w / 2)
        SA = SAB.crop((0, 0, w2, h))
        SB = SAB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        # 对输入和输出进行同样的transform（裁剪也继续采用）
        source_transform_params = get_params(self.opt, SA.size)
        source_A_transform = get_transform(self.opt, source_transform_params, grayscale=(self.input_nc == 1))
        source_B_transform = get_transform(self.opt, source_transform_params, grayscale=(self.output_nc == 1))

        target_transform_params = get_params(self.opt, TA.size, is_source=False)
        target_A_transform = get_transform(self.opt, target_transform_params, grayscale=(self.input_nc == 1))

        SA = source_A_transform(SA)
        # 使用同一个transform
        SA_G = source_A_transform(SA_G)
        SB = source_B_transform(SB)
        TA = target_A_transform(TA)
        TA_G = target_A_transform(TA_G)

        SA = torch.cat([SA, SA_G], dim=0)
        TA = torch.cat([TA, TA_G], dim=0)


        return {'SA': SA, 'SB': SB, 'SA_path': source_path,
                'SB_path': source_path, 'TA': TA, 'TA_path': target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.source_paths)
