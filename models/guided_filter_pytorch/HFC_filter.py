# -*- coding: UTF-8 -*-
"""
@Function:
@File: HFC_filter.py
@Date: 2021/7/26 15:02 
@Author: Hever
"""
from torch import nn
from torch.nn import functional as F
import torch
import cv2


class HFCFilter(nn.Module):
    def __init__(self,
                 # device,
                 filter_width=23, nsig=20, ratio=4, sub_low_ratio=1, sub_mask=False, is_clamp=True):
        super(HFCFilter, self).__init__()
        self.gaussian_filter = Gaussian_kernel(
            # device,
            filter_width, nsig=nsig)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.max = 1.0
        self.min = -1.0
        self.ratio = 4
        self.sub_low_ratio = sub_low_ratio
        self.sub_mask = sub_mask
        self.is_clamp = is_clamp

    # 边缘加上均值
    def median_padding(self, x, mask):
        m_list = []
        batch_size = x.shape[0]
        # x -> torch.Size([1, 3, 256, 256])   i -> [0,2]
        # 计算每个通道的均值  白色是1黑色是0
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2) # 增维 torch.Size([1, 3, 1, 1])
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def forward(self, x, mask):
        assert mask is not None
        x = self.median_padding(x, mask) # 中值填充
        gaussian_output = self.gaussian_filter(x) # 高斯模糊
        res = self.ratio * (x - self.sub_low_ratio * gaussian_output)
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max) # 限制res的value在[self.min, self.max]之间
            # 在input中的任何小于min的值将被设置为min，任何大于max的值将被设置为max。
        if self.sub_mask:
            res = (res + 1) * mask - 1  # 原始[-1,1]  做平移 为了方便显示

        return res

class HFC_LFC_Filter(nn.Module):
    def __init__(self,
                 # device,
                 filter_width=23, nsig=20, ratio=4, sub_low_ratio=1, sub_mask=False, is_clamp=True):
        super(HFC_LFC_Filter, self).__init__()
        self.gaussian_filter = Gaussian_kernel(
            # device,
            filter_width, nsig=nsig)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.max = 1.0
        self.min = -1.0
        self.ratio = 4
        self.sub_low_ratio = sub_low_ratio
        self.sub_mask = sub_mask
        self.is_clamp = is_clamp

    def median_padding(self, x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def forward(self, x, mask):
        assert mask is not None
        x = self.median_padding(x, mask)
        # mask = self.avg_pool(mask)

        gaussian_output = self.gaussian_filter(x)
        res = self.ratio * (x - self.sub_low_ratio * gaussian_output)
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max)
        if self.sub_mask:
            res = (res + 1) * mask - 1

        return res, gaussian_output


def get_kernel(kernel_len=16, nsig=10):  # nsig 标准差 ，kernlen=16核尺寸
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
                     * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel

class Gaussian_kernel(nn.Module):
    def __init__(self,
                 # device,
                 kernel_len, nsig=20):
        super(Gaussian_kernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # 获得高斯卷积核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        kernel = kernel.repeat(3, 1, 1, 1)
        # self.weight = nn.Parameter(images=kernel, requires_grad=False).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len/2))
        # # TODO:为了计算复杂度
        # self.conv = torch.nn.Conv2d(3, 3, kernel_len, 1, int(self.kernel_len / 2))
        # self.conv.weight = self.weight

    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)
        # 对三个channel分别做卷积
        # res = []
        # for i in range(x.shape[1]):
        #     res.append(F.conv2d(x[:, i:i+1], self.weight))
        # x_output = torch.cat(res, dim=1)
        x_output = F.conv2d(x, self.weight, groups=x.shape[1])
        # t = self.conv(x)
        return x_output

