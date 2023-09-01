import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SimulateForDehazeDataset(BaseDataset):
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
        self.dir_source_segmentation = os.path.join(opt.dataroot, 'source_bv_od')  # get the images directory
        self.dir_source_mask = os.path.join(opt.dataroot, 'source_mask')  # get the images directory

        self.source_paths = sorted(make_dataset(self.dir_source, opt.max_dataset_size))  # get images paths
        self.source_segmentation_paths = sorted(make_dataset(self.dir_source_segmentation, opt.max_dataset_size))  # get images paths
        self.source_mask_paths = sorted(make_dataset(self.dir_source_mask, opt.max_dataset_size))  # get images paths

        self.source_usable_size = len(self.dir_source)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        source_path = self.source_paths[index]
        original_image_name = os.path.split(source_path)[-1].split('-')[0]+'.png'
        cup_image_name = os.path.split(source_path)[-1].split('-')[0]+'-0.png'
        source_segmentation_path = os.path.join(self.dir_source_segmentation, original_image_name)
        source_cup_path = os.path.join(self.dir_source_mask, cup_image_name)

        SAB = Image.open(source_path).convert('RGB')
        cup_mask = Image.open(source_cup_path).convert('RGB')
        segmentation_mask = Image.open(source_segmentation_path).convert('RGB')

        w, h = SAB.size
        w2 = int(w / 2)
        SA = SAB.crop((0, 0, w2, h))
        SB = SAB.crop((w2, 0, w, h))

        # 对输入和输出进行同样的transform（裁剪也继续采用）
        source_transform_params = get_params(self.opt, SA.size)
        source_A_transform, source_A_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.input_nc == 1))

        SA = source_A_transform(SA)
        SB = source_A_transform(SB)
        cup_mask = source_A_mask_transform(cup_mask)
        segmentation_mask = source_A_mask_transform(segmentation_mask)


        return {'SA': SA, 'SA_path': source_path,
                'cup_mask': cup_mask,
                'SB': SB, 'segmentation_mask': segmentation_mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.source_paths)
