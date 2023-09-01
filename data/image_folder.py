"""A modified images folder class

We modify the official PyTorch images folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

#它通过遍历列表 IMG_EXTENSIONS 中的每一个后缀，并判断 filename 是否以这个后缀结尾，如果有一个后缀满足条件，那么整个表达式的值就是 True，否则就是 False。
# any()其中一个为true则返回true  全为false则false
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf"), extra_dir=None):
    images = []
    images2 = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir # 判断dir是否为目录

    for root, _, fnames in sorted(os.walk(dir)): # os.walk()深度优先遍历目录
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                if extra_dir is not None:
                    path2 = os.path.join(extra_dir, fname)
                    images2.append(path2)
    if extra_dir is not None:
        return images[:min(max_dataset_size, len(images))], images2[:min(max_dataset_size, len(images))]
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported images extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
