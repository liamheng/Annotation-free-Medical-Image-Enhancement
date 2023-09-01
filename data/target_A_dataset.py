import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class TargetADataset(BaseDataset):
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
        # TODO:修改数据目录
        self.dir_A = os.path.join(opt.dataroot, 'target')  # get the images directory
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get images paths
        # assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
        self.input_nc = 3
        # self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        # split AB images into A and B
        # w, h = A.size
        # w2 = int(w / 2)
        # A = A.crop((0, 0, w2, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        # B = B_transform(B)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
