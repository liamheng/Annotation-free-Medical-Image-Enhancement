import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel, get_transform
from data.image_folder import make_dataset
from PIL import Image


class FuDataset(BaseDataset):
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
        self.dir_source = os.path.join(opt.dataroot, 'source_image')  # get the images directory
        self.dir_source_mask = os.path.join(opt.dataroot, 'source_mask')  # get the images directory
        self.dir_target = os.path.join(opt.dataroot, 'target')  # get the images directory

        self.source_mask_paths = sorted(make_dataset(self.dir_source_mask, opt.max_dataset_size))  # get images paths
        self.source_paths = sorted(make_dataset(self.dir_source, opt.max_dataset_size))  # get images paths
        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get images paths

        self.target_size = len(self.target_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        source_path = self.source_paths[index]
        source_path_mask_path = self.source_mask_paths[index]
        target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]

        SAB = Image.open(source_path).convert('RGB')
        SA_mask = Image.open(source_path_mask_path).convert('L')
        TA = Image.open(target_path).convert('RGB')
        w, h = SAB.size
        w2 = int(w / 2)
        SA = SAB.crop((0, 0, w2, h))
        SB = SAB.crop((w2, 0, w, h))

        # 对输入和输出进行同样的transform（裁剪也继续采用）
        source_transform_params = get_params(self.opt, SA.size)
        source_transform = get_transform(self.opt, source_transform_params, grayscale=(self.input_nc == 1))
        source_A_mask_transform = get_transform(self.opt, source_transform_params, grayscale=(self.output_nc == 1), is_mask=True)

        target_transform_params = get_params(self.opt, TA.size, is_source=False)
        target_A_transform = get_transform(self.opt, target_transform_params, grayscale=(self.input_nc == 1))

        SA = source_transform(SA)
        SB = source_transform(SB)
        SA_mask = source_A_mask_transform(SA_mask)

        TA = target_A_transform(TA)


        return {'input_image': SA, 'source_path': source_path, 'gt_artifact_mask': SA_mask, 'gt_output_image': SB,
                'target': TA, 'target_path': target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.source_paths)
