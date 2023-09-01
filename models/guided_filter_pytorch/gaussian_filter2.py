# -*- coding: UTF-8 -*-
"""
@Function:
@File: gaussian_filter2.py
@Date: 2021/7/8 15:03 
@Author: Hever
"""
from torch import nn
from torch.nn import functional as F
import torch
import cv2


class OneGaussianFilter(nn.Module):
    def __init__(self, device, filter_width=23, size=256, nsig=20):
        super(OneGaussianFilter, self).__init__()
        # self.gaussian_filter = Gaussian_kernel(device, filter_width, nsig=nsig)
        self.gaussian_filter = Gaussian_kernel(device, filter_width, nsig=nsig)
        # TODO：batch_size
        self.ones = torch.ones((1, 1, size, size)).to(device)
        self.minus_ones = -torch.ones((1, 1, size, size)).to(device)

    def median_padding(self, x, mask):
        m_list = []
        # x=torch.cat([x,x])
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            # 略微提高了中值
            # TODO：使用flatten计算batch的median
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
            # m_list.append(x[:, i].median() + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def forward(self, x, mask=None, ratio=4):
        if mask is not None:
            x = self.median_padding(x, mask)

        gaussian_output = self.gaussian_filter(x)
        res = ratio * (x - gaussian_output)
        # up_threshold = torch.quantile(res[:, 0], 0.9)
        res = torch.where(res > 1.0, self.ones, res)
        res = torch.where(res < -1.0, self.minus_ones, res)
        # if mask is not None:
        #     res = res * mask
        return res

class Gaussian_kernel(nn.Module):
    def __init__(self, device, kernel_len, nsig=20):
        super(Gaussian_kernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # 获得高斯卷积核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len/2))


    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)
        # 对三个channel分别做卷积
        res = []
        for i in range(x.shape[1]):
            res.append(F.conv2d(x[:, i:i+1], self.weight))
        x_output = torch.cat(res, dim=1)
        return x_output


def get_kernel(kernel_len=16, nsig=10):  # nsig 标准差 ，kernlen=16核尺寸
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
                     * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel