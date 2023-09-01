import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image


class FiqBasicDataset(BaseDataset):
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
        self.isTrain = opt.isTrain
        # self.need_mask = opt.need_mask
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
        # 判断模型的方向 B->A 意味着 清晰到模糊
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        if self.isTrain:
            image_dir_name = 'source_image' # 生成的模糊图片SB 'source_image'
            gt_dir_name = 'source_gt' # 清晰原图SA 'source_gt'
            image_mask_dir_name = 'source_mask' # 'source_mask'
            gt_mask_dir_name = 'source_mask' # 'source_mask'
        else:
            if self.opt.phase == 'eval':
                image_dir_name = 'eval_image'
                gt_dir_name = 'eval_gt'
                image_mask_dir_name = 'eval_mask'
                gt_mask_dir_name = 'eval_mask'
            elif self.opt.phase == 'drive_test':
                image_dir_name = 'drive_test_image'
                gt_dir_name = 'drive_test_gt'
                image_mask_dir_name = 'drive_test_mask'
                gt_mask_dir_name = 'drive_test_mask'
            elif self.opt.phase == 'avr_test':
                image_dir_name = 'avr_test_image'
                gt_dir_name = 'avr_test_gt'
                image_mask_dir_name = 'avr_test_mask'
                gt_mask_dir_name = 'avr_test_mask'
            else:
                image_dir_name = 'target_image_multi_seg'  # A
                gt_dir_name = 'target_gt'  # B
                image_mask_dir_name = 'target_mask'  # A_mask
                gt_mask_dir_name = 'target_gt_mask'  # A_mask
                # image_dir_name = 'target_image_sat'  # A
                # gt_dir_name = 'target_gt_sat'  # B
                # image_mask_dir_name = 'target_mask'  # A_mask
                # gt_mask_dir_name = 'target_gt_mask'  # A_mask

        self.image_dir = os.path.join(opt.dataroot, image_dir_name)  # get the images directory
        self.gt_dir = os.path.join(opt.dataroot, gt_dir_name)  # get the images directory
        self.image_mask_dir = os.path.join(opt.dataroot, image_mask_dir_name)  # get the images directory
        self.gt_mask_dir = os.path.join(opt.dataroot, gt_mask_dir_name)  # get the images directory

        self.image_paths = sorted(make_dataset(self.image_dir, opt.max_dataset_size))  # get images paths
        # self.gt_paths = sorted(make_dataset(self.gt_dir, opt.max_dataset_size))  # get images paths
        # self.mask_paths = sorted(make_dataset(self.mask_dir, opt.max_dataset_size))  # get images paths


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # 为了适配target
        image_name = os.path.split(image_path)[-1].split('-')[0].replace('.png', '') + '.png'
        gt_path = os.path.join(self.gt_dir, image_name)

        A = Image.open(image_path).convert('RGB')
        B = Image.open(gt_path).convert('RGB')

        # w, h = A.size
        # 对输入和输出进行同样的transform（裁剪也继续采用）
        transform_params = get_params(self.opt, A.size)
        image_transform, mask_transform = get_transform_six_channel(self.opt, transform_params, grayscale=(self.input_nc == 1))

        A = image_transform(A)
        B = image_transform(B)
        # if self.need_mask:
        image_mask_path = os.path.join(self.image_mask_dir, image_name)
        gt_mask_path = os.path.join(self.gt_mask_dir, image_name)
        image_mask = Image.open(image_mask_path).convert('L')
        gt_mask = Image.open(gt_mask_path).convert('L')
        A_mask = mask_transform(image_mask)
        B_mask = mask_transform(gt_mask)
        # return {'A': A, 'B': B, 'A_path': image_path,
        #         'A_mask': A_mask, 'B_mask': B_mask}
        return {'SA': A, 'SB': B, 'S_mask': A_mask, 'SA_path': image_path,
                'SB_path': image_path, 'TA': A, 'T_mask': A_mask, 'TA_path': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
