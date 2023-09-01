# -*- coding: UTF-8 -*-
"""
@Function:
@File: gaussian_filter.py
@Date: 2021/3/28 17:29 
@Author: Hever
"""
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import cv2


# class ThreeGaussianFilter(nn.Module):
#     def __init__(self, device):
#         super(ThreeGaussianFilter, self).__init__()
#         self.gaussian_filter_31 = Gaussian_kernel(device, 31)
#         self.gaussian_filter_41 = Gaussian_kernel(device, 41)
#         self.gaussian_filter_51 = Gaussian_kernel(device, 51)
#
#     def forward(self, x):
#         gaussian_31 = self.gaussian_filter_31(x)
#         gaussian_41 = self.gaussian_filter_41(x)
#         gaussian_51 = self.gaussian_filter_51(x)
#         gaussian_x = torch.cat([gaussian_31, gaussian_41, gaussian_51], dim=1)
#         return gaussian_x


class OneGaussianFilter(nn.Module):
    def __init__(self, device, filter_width=13, size=256, nsig=10):
        super(OneGaussianFilter, self).__init__()
        # self.gaussian_filter = Gaussian_kernel(device, filter_width, nsig=nsig)
        self.gaussian_filter = Gaussian_kernel(device, filter_width, nsig=nsig)
        # TODO：batch_size
        self.ones = torch.ones((1, 1, size, size)).to(device)
        self.minus_ones = -torch.ones((1, 1, size, size)).to(device)

    def median_padding(self):
        pass

    def forward(self, x):
        gaussian_output = self.gaussian_filter(x)
        res = 4 * (x - gaussian_output)
        # ones = torch.ones(x.shape)
        # minus_ones = -torch.ones(x.shape)
        # res = 4 * (x - gaussian_output)
        res = torch.where(res > 1.0, self.ones, res)
        res = torch.where(res < -1.0, self.minus_ones, res)
        return res

class OneGaussianLambdaFilter(nn.Module):
    def __init__(self, device, filter_width=13, size=256, lambda_=0.25):
        super(OneGaussianLambdaFilter, self).__init__()
        self.gaussian_filter = Gaussian_kernel(device, filter_width)
        # TODO：batch_size
        self.ones = torch.ones((1, 1, size, size)).to(device)
        self.minus_ones = -torch.ones((1, 1, size, size)).to(device)
        self.lambda_ = lambda_

    def forward(self, x):
        gaussian_output = self.gaussian_filter(x)
        res = x - self.lambda_ * gaussian_output
        # ones = torch.ones(x.shape)
        # minus_ones = -torch.ones(x.shape)
        # res = torch.where(res > 1.0, self.ones, res)
        # res = torch.where(res < -1.0, self.minus_ones, res)
        return res


class Gaussian_kernel(nn.Module):
    def __init__(self, device, kernel_len, nsig=10):
        super(Gaussian_kernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # 获得高斯卷积核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        # kernel3 = np.expand_dims(kernel3, 0).repeat(3, axis=0)

        self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        # self.BN = nn.BatchNorm2d(num_features=1).to(device)


    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x0 = F.conv2d(x[:, 0:1], self.weight, padding=int(self.kernel_len/2))  # bs,i_c,H1,W1---->bs,1,H1,W1
        x1 = F.conv2d(x[:, 1:2], self.weight, padding=int(self.kernel_len / 2))  # bs,i_c,H1,W1---->bs,1,H1,W1
        x2 = F.conv2d(x[:, 2:3], self.weight, padding=int(self.kernel_len / 2))  # bs,i_c,H1,W1---->bs,1,H1,W1
        x_output = torch.cat([x0, x1, x2], dim=1)
        # x1_attention = self.BN(x1_attention)  # bs,1,H1,W1
        # x_max = torch.max(x1_attention, x1)  # bs,1,H1,W1
        # x_out = x_max * x2  # bs,1,H,W *bs,i_c,H1,W1 =bs,i_c,H_max,W_max (H1和H2取较大的那个)
        return x_output


def get_kernel(kernel_len=16, nsig=10):  # nsig 标准差 ，kernlen=16核尺寸
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
                     * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel


