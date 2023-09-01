import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image


class EyeQSegmentPairDataset(BaseDataset):
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
        self.dir_source_noise = os.path.join(opt.dataroot, 'source_noise')  # get the images directory
        self.dir_source_segmentation = os.path.join(opt.dataroot, 'source_segmentation')  # get the images directory
        self.dir_source_mask = os.path.join(opt.dataroot, 'source_mask')  # get the images directory
        self.dir_source_gt = os.path.join(opt.dataroot, 'source_gt')  # get the images directory


        self.source_noise_paths = sorted(make_dataset(self.dir_source_noise, opt.max_dataset_size))  # get images paths
        self.source_segmentation_paths = sorted(make_dataset(self.dir_source_segmentation, opt.max_dataset_size))  # get images paths
        self.source_mask_paths = sorted(make_dataset(self.dir_source_mask, opt.max_dataset_size))  # get images paths
        self.source_gt_paths = sorted(make_dataset(self.dir_source_gt, opt.max_dataset_size))  # get images paths

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        source_noise_paths = self.source_noise_paths[index]
        original_image_name = os.path.split(source_noise_paths)[-1]
        source_segmentation_path = os.path.join(self.dir_source_segmentation, original_image_name)
        source_mask_path = os.path.join(self.dir_source_mask, os.path.split(source_noise_paths)[-1])
        source_gt_paths = self.source_gt_paths[index]


        source_noise = Image.open(source_noise_paths).convert('RGB')
        mask = Image.open(source_mask_path).convert('RGB')
        # 这是灰度图
        segmentation_mask = Image.open(source_segmentation_path).convert('L')
        source_gt = Image.open(source_gt_paths).convert('RGB')


        # 对输入和输出进行同样的transform（裁剪也继续采用）
        source_transform_params = get_params(self.opt, source_noise.size)
        # TODO:transform
        source_A_transform, source_A_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.input_nc == 1))
        # target_transform_params = get_params(self.opt, SU.size, is_source=False)
        # target_B_transform = get_transform(self.opt, target_transform_params, grayscale=(self.input_nc == 1))

        source_noise = source_A_transform(source_noise)
        mask = source_A_mask_transform(mask)
        segmentation_mask = source_A_mask_transform(segmentation_mask)

        # TODO label是否也需要transform
        # 将source_gt转换为tensor
        source_gt = source_A_transform(source_gt)




        return {'source_noise': source_noise, 'source_gt': source_gt,
                'mask': mask, 'segmentation_mask': segmentation_mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.source_noise_paths)
