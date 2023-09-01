import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image


class FIQDataset(BaseDataset):
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
        self.dir_source_noise = os.path.join(opt.dataroot, 'input')  # get the images directory
        self.dir_source_gt = os.path.join(opt.dataroot, 'gt')  # get the images directory
        self.dir_source_mask = os.path.join(opt.dataroot, 'gt_mask')  # get the images directory
        self.input_mask = os.path.join(opt.dataroot, 'input_mask')  # get the images directory

        # self.source_noise_paths = sorted(make_dataset(self.dir_source_noise, opt.max_dataset_size))  # get images paths
        # self.source_gt_paths = sorted(make_dataset(self.dir_source_gt, opt.max_dataset_size))  # get images paths
        # self.source_mask_paths = sorted(make_dataset(self.dir_source_mask, opt.max_dataset_size))  # get images paths
        # self.input_mask_paths = sorted(make_dataset(self.input_mask, opt.max_dataset_size))  # get images paths
        self.image_paths = sorted(make_dataset(self.dir_source_noise, opt.max_dataset_size))  # get images paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded images

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain# get images paths


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # 为了适配target
        image_name = os.path.split(image_path)[-1].split('-')[0].replace('.png', '') + '.png'
        # gt_path = self.image_paths[random.randint(0, len(self.image_paths) - 1)].split('-')[0].replace('.png',
        #                                                                                                '').replace(
        #     'source_image', 'source_gt') + '.png'

        gt_path = os.path.join(self.dir_source_gt, image_name)
        A = Image.open(image_path).convert('RGB')
        B = Image.open(gt_path).convert('RGB')

        # w, h = A.size
        # 对输入和输出进行同样的transform（裁剪也继续采用）
        transform_params = get_params(self.opt, A.size)
        image_transform, mask_transform = get_transform_six_channel(self.opt, transform_params,
                                                                    grayscale=(self.input_nc == 1))

        source_noise = image_transform(A)
        source_gt = image_transform(B)
        # if self.need_mask:
        image_mask_path = os.path.join(self.input_mask, image_name)
        gt_mask_path = os.path.join(self.dir_source_mask, image_name)
        image_mask = Image.open(image_mask_path).convert('L')
        gt_mask = Image.open(gt_mask_path).convert('L')
        image_mask = mask_transform(image_mask)
        gt_mask = mask_transform(gt_mask)

        # return {'source_noise': source_noise, 'source_gt': source_gt, "SA_path": image_path, "mask": gt_mask,
        #         "input_mask" : image_mask}

        return {'TA': source_noise, 'source_gt': source_gt, "SA_path": image_path,
            "mask": image_mask, "segmentation_mask": image_mask, "TA_path": image_path, "T_mask": image_mask}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
