"""
先运行test获取运行结果，此处运行需要输入模型的输出文件夹
在对应的target_gt中取mask（512），mask resize到256，应用在fake TB上，并且保存下来
target_gt再resize到512，之后就能开始计算SSIM
"""
import os
import torch
import lpips
import cv2
import numpy as np
import argparse
from options.test_options import TestOptions
from model_eval.fid_score import get_is_fid_score, get_is_score
from data.cataract_test_dataset import CataractTestDataset
# from metrics.brisque import *
# from joblib import load
from model_eval.metrics.brisque import brisque
# from model_eval.metrics.niqe.niqe import niqe
from model_eval.metrics.niqe2.niqe import niqe
from scipy import ndimage


def eval_public(opt, cataract_test_dataset):
    # get_is_fid(opt, cataract_test_dataset)
    # get_is(opt, cataract_test_dataset)
    # get_brisque(opt, cataract_test_dataset)
    get_niqe(opt, cataract_test_dataset)


def get_is_fid(opt, cataract_test_dataset):
    cataract_test_dataset.mode = 1
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    is_mean, is_std, fid = get_is_fid_score(cataract_test_dataset, device=device, dataroot=opt.dataroot)
    with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        f.write('%f,%f,%f,' % (is_mean, is_std, fid))
    # return is_mean, is_std, fid


def get_is(opt, cataract_test_dataset):
    cataract_test_dataset.mode = 1
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    is_mean, is_std = get_is_score(cataract_test_dataset, device=device)
    with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        f.write('%f,%f,' % (is_mean, is_std))
    # return is_mean, is_std, fid

def get_lpips(opt, cataract_test_dataset, mode=2):
    # 让两张图像相似，不采用
    cataract_test_dataset.mode = mode
    loss_fn_alex = lpips.LPIPS(net='alex')
    sum_lpips = 0
    for i, batch in enumerate(cataract_test_dataset):
        # batchv = Variable(batch)
        # batch_size_i = batch.size()[0]
        img0, img1 = batch
        d = loss_fn_alex(img0, img1)
        sum_lpips += d
        print(d)
    print(sum_lpips, sum_lpips / len(cataract_test_dataset))
    with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        f.write('%f,' % (sum_lpips / len(cataract_test_dataset)))


def get_brisque(opt, cataract_test_dataset, mode=4):
    # mode=4
    cataract_test_dataset.mode = mode
    sum_brisque = 0.0
    dataloader = torch.utils.data.DataLoader(cataract_test_dataset,
        batch_size=8,shuffle=False)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    for i, images in enumerate(dataloader):
        # batchv = Variable(batch)
        # batch_size_i = batch.size()[0]
        images = images.to(device)
        score = brisque(images, reduction='sum')
        sum_brisque += score.cpu().numpy()
    print('brisque:', sum_brisque / len(cataract_test_dataset))
    with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        f.write('%f,' % (sum_brisque / len(cataract_test_dataset)))
    return sum_brisque / len(cataract_test_dataset)


def mul_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = ndimage.binary_opening(gray > 10, structure=np.ones((8, 8)))

    image = np.transpose(image, (2, 0, 1))
    image = image * mask
    image = np.transpose(image, (1, 2, 0))

    return image

def get_niqe(opt, cataract_test_dataset, mode=5):
    cataract_test_dataset.mode = mode
    sum_niqe = 0.0
    rec = {}
    for i, B_path in enumerate(cataract_test_dataset):
        # batchv = Variable(batch)
        # batch_size_i = batch.size()[0]
        image = cv2.imread(B_path)
        if cataract_test_dataset.mul_mask:
            image = mul_mask(image)
        score = niqe(image)
        sum_niqe += score
        rec[B_path] = score

    print('sum_niqe:', sum_niqe / len(cataract_test_dataset))
    with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        f.write('%f,\n' % (sum_niqe / len(cataract_test_dataset)))
    return sum_niqe / len(cataract_test_dataset)


def eval_after_training(opt, max_epoch=150, step=5, mul_mask=False):
    # for epoch in range(5, max_epoch+1, step):
    test_web_dir = os.path.join(opt.results_dir, opt.name,
                                '{}_{}'.format(opt.phase, 'latest'))
    test_web_dir = '{:s}_iter{:d}'.format(test_web_dir, opt.load_iter)
    print(test_web_dir)

    cataractTestDataset = CataractTestDataset(opt, test_web_dir, mul_mask=mul_mask)
    eval_public(opt, cataractTestDataset)


class EvalOptions(TestOptions):
    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--mul_mask', action='store_true')
        return parser


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('create images pairs')
    # parser.add_argument('--mul_mask', action='store_true')
    opt = EvalOptions().parse()  # get test_total options
    # opt = TestOptions.

    # cataract_test_dataset = CataractTestDataset(opt, mode=2)
    # device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    # cataract_test_dataset = CataractTestDataset(opt, mode=3)
    # get_lpips(cataract_test_dataset)
    # get_brisque(cataract_test_dataset)
    # get_niqe(opt, cataract_test_dataset)
    eval_after_training(opt, mul_mask=True)