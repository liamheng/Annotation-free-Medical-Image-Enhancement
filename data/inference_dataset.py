import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image
import cv2
from scipy import ndimage
import numpy as np
from PIL import Image

def get_mask(img):
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return ndimage.binary_opening(gray > 5, structure=np.ones((8, 8)))

class FiqInferenceDataset(BaseDataset):
    """A dataset class for paired images dataset.

    It assumes that the directory '/path/to/images/train' contains images pairs in the form of {A,B}.
    During test_total time, you need to prepare a directory '/path/to/images/test_total'.
    """

#     def __init__(self, opt):
#         """Initialize this dataset class.
#
#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)
#         # self.dir_source = os.path.join(opt.dataroot, 'source')  # get the images directory
#         self.dir_target = os.path.join(opt.dataroot, 'target')  # get the images directory
#         # self.dir_source_mask = os.path.join(opt.dataroot, 'source_mask')  # get the images directory
#         self.dir_target_mask = os.path.join(opt.dataroot, 'target_mask')  #
#
#         # self.source_paths = sorted(make_dataset(self.dir_source, opt.max_dataset_size))  # get images paths
#         self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get images paths
#         # self.source_mask_paths = sorted(make_dataset(self.dir_source_mask, opt.max_dataset_size))  # get images paths
#         self.target_mask_paths = sorted(make_dataset(self.dir_target_mask, opt.max_dataset_size))  # get images paths
#
#         self.target_size = len(self.target_paths)
#         assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
#
#         self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
#         self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
#         self.isTrain = opt.isTrain
#
#     def __getitem__(self, index):
#         """Return a images point and its metadata information.
#
#         Parameters:
#             index - - a random integer for images indexing
#
#         Returns a dictionary that contains A, B, A_paths and B_paths
#             A (tensor) - - an images in the input domain
#             B (tensor) - - its corresponding images in the target domain
#             A_paths (str) - - images paths
#             B_paths (str) - - images paths (same as A_paths)
#         """
#         # read a images given a random integer index
#         # source_path = self.source_paths[index]
#         # source_path_mask_path = os.path.join(self.dir_source_mask, os.path.split(source_path)[-1].replace('jpg', 'png'))
#             # self.source_mask_paths[index]
#         target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
#         # print(target_index)
#         target_path = self.target_paths[target_index]
#         target_mask_path = self.target_mask_paths[target_index]
#
#         # SAB = Image.open(source_path).convert('RGB')
#         TA = Image.open(target_path).convert('RGB')
#         T_mask = get_mask(TA)
#         T_mask = T_mask
#         T_mask = T_mask.astype(np.int8)
#         # T_mask = T_mask.astype(np.int8) * 255
#         # T_mask = T_mask[np.newaxis, :, :]
#         T_mask = Image.fromarray(T_mask)
#
#         # pass
#
#         # SA_mask = Image.open(source_path_mask_path).convert('L')
#         # SB_mask = SA_mask
#         # TA_mask = Image.open(target_mask_path).convert('L')
#         # w, h = SAB.size
#         # w2 = int(w / 2)
#         # SA = SAB.crop((0, 0, w2, h))
#         # SB = SAB.crop((w2, 0, w, h))
#
#
#         # 对输入和输出进行同样的transform（裁剪也继续采用）
#         # source_transform_params = get_params(self.opt, SA.size)
#         # source_A_transform, source_A_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.input_nc == 1))
#         # source_B_transform, source_B_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.output_nc == 1))
#         #  = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.input_nc == 1))
#         # source_B_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.output_nc == 1))
#
#         target_transform_params = get_params(self.opt, TA.size, is_source=False)
#         target_A_transform, target_A_mask_transform = get_transform_six_channel(self.opt, target_transform_params, grayscale=(self.input_nc == 1))
#
#         # SA = source_A_transform(SA)
#         # S_mask = source_A_mask_transform(SA_mask)
#         # # 使用同一个transform
#         # SB = source_B_transform(SB)
#         # SB_mask = source_B_mask_transform(SB_mask)
#
#         TA = target_A_transform(TA)
#         # T_mask = torch.tensor(T_mask).unsqueeze(0)
#         T_mask = target_A_mask_transform(T_mask)
#
#         return {'SA': TA, 'SB': TA, 'S_mask': T_mask, 'SA_path': target_path,
#                 'SB_path': target_path, 'TA': TA, 'T_mask': T_mask, 'TA_path': target_path}
#
#     def __len__(self):
#         """Return the total number of images in the dataset."""
#         return len(self.source_paths)

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.need_mask = opt.need_mask
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded images
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        image_dir_name = 'target'  # A
        image_mask_dir_name = 'target_mask'  # A_mask

        self.image_dir = os.path.join(opt.dataroot, image_dir_name)  # get the images directory
        self.image_mask_dir = os.path.join(opt.dataroot, image_mask_dir_name)  # get the images directory
        # TODO:专门测试多疾病数据集ODIR
        # self.image_dir = '/images/liuhaofeng/Project/multi_disease_classifier/dataset/multi_diseases/image_degraded_3_256'
        # self.image_mask_dir = './datasets/ODIR/target_mask'

        self.image_paths = sorted(make_dataset(self.image_dir, opt.max_dataset_size))  # get images paths
        # self.gt_paths = sorted(make_dataset(self.gt_dir, opt.max_dataset_size))  # get images paths
        # self.mask_paths = sorted(make_dataset(self.mask_dir, opt.max_dataset_size))  # get images paths


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # 为了适配target
        image_name = os.path.split(image_path)[-1].split('-')[0].replace('.png', '') + '.png'

        A = Image.open(image_path).convert('RGB')

        # w, h = A.size
        # 对输入和输出进行同样的transform（裁剪也继续采用）
        # transform_params = get_params(self.opt, A.size)
        transform_params = get_params(self.opt, A.size, is_source=False)
        image_transform, mask_transform = get_transform_six_channel(self.opt, transform_params, grayscale=(self.input_nc == 1))
        T_mask = get_mask(A)
        # T_mask = T_mask
        T_mask = T_mask.astype(np.int8)
        # T_mask = T_mask.astype(np.int8) * 255
        # T_mask = T_mask[np.newaxis, :, :]
        T_mask = Image.fromarray(T_mask)

        A = image_transform(A)
        T_mask = mask_transform(T_mask)

        # if self.need_mask:
        #
        #     image_mask_path = os.path.join(self.image_mask_dir, image_name)
        #     image_mask = Image.open(image_mask_path).convert('L')
        #     A_mask = mask_transform(image_mask)
        #
        #     return {'SA': A, 'SB': A, 'S_mask': A_mask, 'SA_path': image_path,
        #         'SB_path': image_path, 'TA': A, 'T_mask': A_mask, 'TA_path': image_path}
        # A_mask = torch.ones(size=[1,A.shape[1],A.shape[2]])
        return {'SA': A, 'SB': A, 'S_mask': T_mask, 'SA_path': image_path,
                'SB_path': image_path, 'TA': A, 'T_mask': T_mask, 'TA_path': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
