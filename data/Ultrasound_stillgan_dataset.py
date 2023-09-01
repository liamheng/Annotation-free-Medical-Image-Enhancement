import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class UltrasoundStillganDataset(BaseDataset):
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
        # self.dir_sourceA = os.path.join(opt.dataroot, 'three_low')  # get the images directory
        # self.dir_sourceB = os.path.join(opt.dataroot, 'three_high')  # get the images directory
        # self.dir_target = os.path.join(opt.dataroot, 'three_low')  # get the images directory
        # self.dir_sourceA = os.path.join(opt.dataroot, 'two_low')  # get the images directory
        # self.dir_sourceB = os.path.join(opt.dataroot, 'two_high')  # get the images directory
        # self.dir_target = os.path.join(opt.dataroot, 'two_low')  # get the images directory
        # self.dir_sourceA = os.path.join(opt.dataroot, 'all_low')  # get the images directory
        # self.dir_sourceB = os.path.join(opt.dataroot, 'all_high')  # get the images directory
        # self.dir_target = os.path.join(opt.dataroot, 'all_low')  # get the images directory
        self.dir_sourceA = os.path.join(opt.dataroot, 'test')  # get the images directory
        self.dir_sourceB = os.path.join(opt.dataroot, 'test')  # get the images directory
        self.dir_target = os.path.join(opt.dataroot, 'test')  # get the images directory
        self.sourceA_paths = sorted(make_dataset(self.dir_sourceA, opt.max_dataset_size))  # get images paths
        self.sourceB_paths = sorted(make_dataset(self.dir_sourceB, opt.max_dataset_size))  # get images paths
        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get images paths
        self.target_size = len(self.target_paths)
        self.source_size = len(self.sourceA_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain


    def __getitem__(self, index):
        # read a images given a random integer index
        sourceA_path = self.sourceA_paths[index]
        # sourceB_index = random.randint(0, self.source_size - 1) if self.isTrain else index % self.source_size
        # sourceB_path = self.sourceB_paths[sourceB_index]
        sourceB_path = self.sourceB_paths[index]
        SA = Image.open(sourceA_path).convert('L')
        SB = Image.open(sourceB_path).convert('L')


        # apply the same transform to both A and B
        # 对输入和输出进行同样的transform（裁剪也继续采用）
        sourceA_transform_params = get_params(self.opt, SA.size)
        sourceA_transform = get_transform(self.opt, sourceA_transform_params, grayscale=(self.input_nc == 1))
        # TODO:可以给target做一个独立的transform
        sourceB_transform_params = get_params(self.opt, SB.size)
        sourceB_transform = get_transform(self.opt, sourceB_transform_params, grayscale=(self.input_nc == 1))

        SA = sourceA_transform(SA)
        SB = sourceA_transform(SB)
        # SB = sourceB_transform(SB)

        # if not self.isTrain:
        target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]
        TA = Image.open(target_path).convert('L')
        target_transform_params = get_params(self.opt, TA.size, is_source=False)
        target_A_transform = get_transform(self.opt, target_transform_params, grayscale=(self.input_nc == 1))
        TA = target_A_transform(TA)
        return {'SA': SA, 'SB': SB, 'SA_path': sourceA_path, 'SB_path': sourceB_path, 'TA': TA, 'TA_path': target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.isTrain:
            return len(self.sourceA_paths)
        else:
            return len(self.target_paths)
