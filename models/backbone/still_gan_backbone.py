# -*- coding: UTF-8 -*-
"""
@Function:
@File: still_gan_backbone.py
@Date: 2022/7/15 16:47 
@Author: Hever
"""
import functools
import math

import numpy as np
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            norm_layer(ch_out),
        )
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)

        return self.relu(out + residual)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, use_dropout=False):
        super(up_conv,self).__init__()
        if use_dropout:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.5)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.up(x)

        return x

class ResUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(2)

        self.Conv1 = res_conv_block(ch_in=img_ch, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]
        self.Conv2 = res_conv_block(ch_in=ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]
        self.Conv3 = res_conv_block(ch_in=2 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]
        self.Conv4 = res_conv_block(ch_in=4 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 8, W / 8]
        self.Conv5 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 16, W / 16]
        self.Conv6 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 32, W / 32]
        self.Conv7 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 64, W / 64]
        self.Conv8 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up8 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 128, W / 128]
        self.Up_conv8 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up7 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 64, W / 64]
        self.Up_conv7 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 64, W / 64]

        self.Up6 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 32, W / 32]
        self.Up_conv6 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 32, W / 32]

        self.Up5 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 16, W / 16]
        self.Up_conv5 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 16, W / 16]

        self.Up4 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 8, W / 8]
        self.Up_conv4 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 8, W / 8]

        self.Up3 = up_conv(ch_in=8 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]
        self.Up_conv3 = res_conv_block(ch_in=8 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]

        self.Up2 = up_conv(ch_in=4 * ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]
        self.Up_conv2 = res_conv_block(ch_in=4 * ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]

        self.Up1 = up_conv(ch_in=2 * ngf, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]
        self.Up_conv1 = res_conv_block(ch_in=2 * ngf, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]

        self.Conv_1x1 = nn.Conv2d(ngf, output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # [B, ngf, H, W]

        x2 = self.Maxpool(x1)  # [B, ngf, H / 2, W / 2]
        x2 = self.Conv2(x2)  # [B, 2 * ngf, H / 2, W / 2]

        x3 = self.Maxpool(x2)  # [B, 2 * ngf, H / 4, W / 4]
        x3 = self.Conv3(x3)  # [B, 4 * ngf, H / 4, W / 4]

        x4 = self.Maxpool(x3)  # [B, 4 * ngf, H / 8, W / 8]
        x4 = self.Conv4(x4)  # [B, 8 * ngf, H / 8, W / 8]

        x5 = self.Maxpool(x4)  # [B, 8 * ngf, H / 16, W / 16]
        x5 = self.Conv5(x5)  # [B, 8 * ngf, H / 16, W / 16]

        x6 = self.Maxpool(x5)  # [B, 8 * ngf, H / 32, W / 32]
        x6 = self.Conv6(x6)  # [B, 8 * ngf, H / 32, W / 32]

        x7 = self.Maxpool(x6)  # [B, 8 * ngf, H / 64, W / 64]
        x7 = self.Conv7(x7)  # [B, 8 * ngf, H / 64, W / 64]

        x8 = self.Maxpool(x7)  # [B, 8 * ngf, H / 128, W / 128]
        x8 = self.Conv8(x8)  # [B, 8 * ngf, H / 128, W / 128]

        x9 = self.Maxpool(x8)  # [B, 8 * ngf, H / 256, W / 256]

        # decoding + concat path
        d8 = self.Up8(x9)  # [B, 8 * ngf, H / 128, W / 128]
        d8 = torch.cat((x8, d8), dim=1)  # [B, 16 * ngf, H / 128, W / 128]
        d8 = self.Up_conv8(d8)  # [B, 8 * ngf, H / 128, W / 128]

        d7 = self.Up7(d8)  # [B, 8 * ngf, H / 64, W / 64]
        d7 = torch.cat((x7, d7), dim=1)  # [B, 16 * ngf, H / 64, W / 64]
        d7 = self.Up_conv7(d7)  # [B, 8 * ngf, H / 64, W / 64]

        d6 = self.Up6(d7)  # [B, 8 * ngf, H / 32, W / 32]
        d6 = torch.cat((x6, d6), dim=1)  # [B, 16 * ngf, H / 32, W / 32]
        d6 = self.Up_conv6(d6)  # [B, 8 * ngf, H / 32, W / 32]

        d5 = self.Up5(d6)  # [B, 8 * ngf, H / 16, W / 16]
        d5 = torch.cat((x5, d5), dim=1)  # [B, 16 * ngf, H / 16, W / 16]
        d5 = self.Up_conv5(d5)  # [B, 8 * ngf, H / 16, W / 16]

        d4 = self.Up4(d5)  # [B, 8 * ngf, H / 8, W / 8]
        d4 = torch.cat((x4, d4), dim=1)  # [B, 16 * ngf, H / 8, W / 8]
        d4 = self.Up_conv4(d4)  # [B, 8 * ngf, H / 8, W / 8]

        d3 = self.Up3(d4)  # [B, 4 * ngf, H / 4, W / 4]
        d3 = torch.cat((x3, d3), dim=1)  # [B, 8 * ngf, H / 4, W / 4]
        d3 = self.Up_conv3(d3)  # [B, 4 * ngf, H / 4, W / 4]

        d2 = self.Up2(d3)  # [B, 2 * ngf, H / 2, W / 2]
        d2 = torch.cat((x2, d2), dim=1)  # [B, 4 * ngf, H / 2, W / 2]
        d2 = self.Up_conv2(d2)  # [B, 2 * ngf, H / 2, W / 2]

        d1 = self.Up1(d2)  # [B, ngf, H, W]
        d1 = torch.cat((x1, d1), dim=1)  # [B, 2 * ngf, H, W]
        d1 = self.Up_conv1(d1)  # [B, ngf, H, W]

        out = nn.Tanh()(self.Conv_1x1(d1))

        return out




class DeepResUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DeepResUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(2)

        self.Conv1 = res_conv_block(ch_in=img_ch, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]
        self.Conv2 = res_conv_block(ch_in=ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]
        self.Conv3 = res_conv_block(ch_in=2 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]
        self.Conv4 = res_conv_block(ch_in=4 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 8, W / 8]
        self.Conv5 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf,
                                    norm_layer=norm_layer)  # [B, 8 * ngf, H / 16, W / 16]
        self.Conv6 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf,
                                    norm_layer=norm_layer)  # [B, 8 * ngf, H / 32, W / 32]
        self.Conv7 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf,
                                    norm_layer=norm_layer)  # [B, 8 * ngf, H / 64, W / 64]
        self.Conv8 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf,
                                    norm_layer=norm_layer)  # [B, 8 * ngf, H / 128, W / 128]
        self.Conv9 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf,
                                    norm_layer=norm_layer)  # [B, 8 * ngf, H / 256, W / 256]
        self.Conv10 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf,
                                     norm_layer=norm_layer)  # [B, 8 * ngf, H / 256, W / 256]

        self.Up9 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 256, W / 256]
        self.Up_conv9 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf,
                                       norm_layer=norm_layer)  # [B, 8 * ngf, H / 256, W / 256]

        self.Up10 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 256, W / 256]
        self.Up_conv10 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf,
                                        norm_layer=norm_layer)  # [B, 8 * ngf, H / 256, W / 256]

        self.Up8 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout)  # [B, 8 * ngf, H / 128, W / 128]
        self.Up_conv8 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf,
                                       norm_layer=norm_layer)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up7 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout)  # [B, 8 * ngf, H / 64, W / 64]
        self.Up_conv7 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf,
                                       norm_layer=norm_layer)  # [B, 8 * ngf, H / 64, W / 64]

        self.Up6 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout)  # [B, 8 * ngf, H / 32, W / 32]
        self.Up_conv6 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf,
                                       norm_layer=norm_layer)  # [B, 8 * ngf, H / 32, W / 32]

        self.Up5 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout)  # [B, 8 * ngf, H / 16, W / 16]
        self.Up_conv5 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf,
                                       norm_layer=norm_layer)  # [B, 8 * ngf, H / 16, W / 16]

        self.Up4 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout)  # [B, 8 * ngf, H / 8, W / 8]
        self.Up_conv4 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf,
                                       norm_layer=norm_layer)  # [B, 8 * ngf, H / 8, W / 8]

        self.Up3 = up_conv(ch_in=8 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]
        self.Up_conv3 = res_conv_block(ch_in=8 * ngf, ch_out=4 * ngf,
                                       norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]

        self.Up2 = up_conv(ch_in=4 * ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]
        self.Up_conv2 = res_conv_block(ch_in=4 * ngf, ch_out=2 * ngf,
                                       norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]

        self.Up1 = up_conv(ch_in=2 * ngf, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]
        self.Up_conv1 = res_conv_block(ch_in=2 * ngf, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]

        self.Conv_1x1 = nn.Conv2d(ngf, output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # [B, ngf, H, W]

        x2 = self.Maxpool(x1)  # [B, ngf, H / 2, W / 2]
        x2 = self.Conv2(x2)  # [B, 2 * ngf, H / 2, W / 2]

        x3 = self.Maxpool(x2)  # [B, 2 * ngf, H / 4, W / 4]
        x3 = self.Conv3(x3)  # [B, 4 * ngf, H / 4, W / 4]

        x4 = self.Maxpool(x3)  # [B, 4 * ngf, H / 8, W / 8]
        x4 = self.Conv4(x4)  # [B, 8 * ngf, H / 8, W / 8]

        x5 = self.Maxpool(x4)  # [B, 8 * ngf, H / 16, W / 16]
        x5 = self.Conv5(x5)  # [B, 8 * ngf, H / 16, W / 16]

        x6 = self.Maxpool(x5)  # [B, 8 * ngf, H / 32, W / 32]
        x6 = self.Conv6(x6)  # [B, 8 * ngf, H / 32, W / 32]

        x7 = self.Maxpool(x6)  # [B, 8 * ngf, H / 64, W / 64]
        x7 = self.Conv7(x7)  # [B, 8 * ngf, H / 64, W / 64]

        x8 = self.Maxpool(x7)  # [B, 8 * ngf, H / 128, W / 128]
        x8 = self.Conv8(x8)  # [B, 8 * ngf, H / 128, W / 128]

        x9 = self.Maxpool(x8)  # [B, 8 * ngf, H / 256, W / 256]
        x9 = self.Conv9(x9)  # [B, 8 * ngf, H / 256, W / 256]

        x10 = self.Maxpool(x9)  # [B, 8 * ngf, H / 256, W / 256]
        x10 = self.Conv10(x10)  # [B, 8 * ngf, H / 256, W / 256]

        x11 = self.Maxpool(x10)  # [B, 8 * ngf, H / 256, W / 256]

        # decoding + concat path
        d10 = self.Up10(x11)  # [B, 8 * ngf, H / 256, W / 256]
        d10 = torch.cat((x10, d10), dim=1)  # [B, 16 * ngf, H / 256, W / 256]
        d10 = self.Up_conv10(d10)  # [B, 8 * ngf, H / 256, W / 256]



        d9 = self.Up9(d10)  # [B, 8 * ngf, H / 256, W / 256]
        d9 = torch.cat((x9, d9), dim=1)  # [B, 16 * ngf, H / 256, W / 256]
        d9 = self.Up_conv9(d9)  # [B, 8 * ngf, H / 256, W / 256]

        # decoding + concat path
        d8 = self.Up8(d9)  # [B, 8 * ngf, H / 128, W / 128]
        d8 = torch.cat((x8, d8), dim=1)  # [B, 16 * ngf, H / 128, W / 128]
        d8 = self.Up_conv8(d8)  # [B, 8 * ngf, H / 128, W / 128]

        d7 = self.Up7(d8)  # [B, 8 * ngf, H / 64, W / 64]
        d7 = torch.cat((x7, d7), dim=1)  # [B, 16 * ngf, H / 64, W / 64]
        d7 = self.Up_conv7(d7)  # [B, 8 * ngf, H / 64, W / 64]

        d6 = self.Up6(d7)  # [B, 8 * ngf, H / 32, W / 32]
        d6 = torch.cat((x6, d6), dim=1)  # [B, 16 * ngf, H / 32, W / 32]
        d6 = self.Up_conv6(d6)  # [B, 8 * ngf, H / 32, W / 32]

        d5 = self.Up5(d6)  # [B, 8 * ngf, H / 16, W / 16]
        d5 = torch.cat((x5, d5), dim=1)  # [B, 16 * ngf, H / 16, W / 16]
        d5 = self.Up_conv5(d5)  # [B, 8 * ngf, H / 16, W / 16]

        d4 = self.Up4(d5)  # [B, 8 * ngf, H / 8, W / 8]
        d4 = torch.cat((x4, d4), dim=1)  # [B, 16 * ngf, H / 8, W / 8]
        d4 = self.Up_conv4(d4)  # [B, 8 * ngf, H / 8, W / 8]

        d3 = self.Up3(d4)  # [B, 4 * ngf, H / 4, W / 4]
        d3 = torch.cat((x3, d3), dim=1)  # [B, 8 * ngf, H / 4, W / 4]
        d3 = self.Up_conv3(d3)  # [B, 4 * ngf, H / 4, W / 4]

        d2 = self.Up2(d3)  # [B, 2 * ngf, H / 2, W / 2]
        d2 = torch.cat((x2, d2), dim=1)  # [B, 4 * ngf, H / 2, W / 2]
        d2 = self.Up_conv2(d2)  # [B, 2 * ngf, H / 2, W / 2]

        d1 = self.Up1(d2)  # [B, ngf, H, W]
        d1 = torch.cat((x1, d1), dim=1)  # [B, 2 * ngf, H, W]
        d1 = self.Up_conv1(d1)  # [B, ngf, H, W]
        out = self.Conv_1x1(d1)
        # out = nn.Tanh()(self.Conv_1x1(d1))

        return out

# 测试DeepResUNet
if __name__ == '__main__':
    net = ResUNet(1, 1)
     # 生成输入数据
    input = torch.randn(1, 1, 512, 512)
    out = net(input)
    print(out.shape)
class StructureLoss(nn.Module):
    """Define Structure Loss.

    Structure Loss reflects the structural difference between inputs and outputs to some extent.
    """

    def __init__(self, channel=1, window_size=11, crop_size=384, size_average=True):
        """Initialize the StructureLoss class.

        Parameters:
            channel (int) - - number of channels
            window_size (int) - - size of window
            size_average (bool) - - average of batch or not
        """
        super(StructureLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.crop_size = crop_size
        self.window = create_window(window_size, channel)

    def forward(self, img1, img2):
        (_, channel, height, width) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window.data = window
            self.channel = channel

        inputs1 = (img1[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0
        inputs2 = (img2[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0

        return 1.0 - _ssim(inputs1, inputs2, window, self.window_size, channel, self.size_average)

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = nn.Parameter(data=_2D_window.expand(channel, 1, window_size, window_size).contiguous(), requires_grad=False)
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _ssim(img1, img2, window, window_size, channel, K1=0.01, K2=0.03, L=1.0, size_average=True):
    assert img1.size() == img2.size()
    noise = torch.Tensor(np.random.normal(0, 0.01, img1.size())).cuda(img1.get_device())
    new_img1 = clip_by_tensor(img1 + noise, torch.Tensor(np.zeros(img1.size())).cuda(img1.get_device()), torch.Tensor(np.ones(img1.size())).cuda(img1.get_device()))
    new_img2 = clip_by_tensor(img2 + noise, torch.Tensor(np.zeros(img2.size())).cuda(img2.get_device()), torch.Tensor(np.ones(img2.size())).cuda(img2.get_device()))
    mu1 = F.conv2d(new_img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(new_img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(new_img1 * new_img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(new_img2 * new_img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(new_img1 * new_img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2.0

    ssim_map = (sigma12 + C3) / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C3)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result



class LuminanceLoss(nn.Module):
    """Define Illumination Regularization.

    Illumination Regularization reflects the degree of inputs' uneven illumination to some extent.
    """

    def __init__(self, patch_height, patch_width, crop_size=384):
        """Initialize the LuminanceLoss class.

        Parameters:
            patch_height (int) - - height of patch
            patch_width (int) - - width of patch
        """
        super(LuminanceLoss, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.crop_size = crop_size
        self.avgpool = nn.AvgPool2d((patch_height, patch_width), stride=(patch_height, patch_width))

    def forward(self, inputs):
        height = inputs.size()[2]
        width = inputs.size()[3]
        assert height >= self.crop_size and width >= self.crop_size
        assert self.crop_size % self.patch_height == 0 and self.crop_size % self.patch_width == 0, "Patch size Error."

        crop_inputs = (inputs[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0
        # [batch_size, channels, self.crop_size, self.crop_size] --> [batch_size, channels, 1, 1]
        global_mean = torch.mean(crop_inputs, [2, 3], True)
        # [batch_size, channels, self.crop_size, self.crop_size] --> [batch_size, channels, N, M]
        D = self.avgpool(crop_inputs)
        E = D - global_mean.expand_as(D)  # [batch_size, channels, N, M]
        upsample = nn.Upsample(size=[self.crop_size, self.crop_size], mode='bicubic', align_corners=False)
        R = upsample(E)  # [batch_size, channels, self.crop_size, self.crop_size]

        return torch.abs(R).mean()



class NCCLoss(nn.Module):
    def __init__(self, win=None):
        super(NCCLoss, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):
        I = y_true
        J = y_pred

        ndims = len(I.size()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        win = [9] * ndims if self.win is None else self.win

        sum_filt = torch.ones([1, 1, *win]).to(I.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)

# class NCC:
#     """
#     Local (over window) normalized cross correlation loss.
#     """
#
#     def __init__(self, win=None):
#         self.win = win
#
#     def loss(self, y_true, y_pred):
#
#         I = y_true
#         J = y_pred
#
#         # get dimension of volume
#         # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
#         ndims = len(list(I.size())) - 2
#         assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
#
#         # set window size
#         win = [9] * ndims if self.win is None else self.win
#
#         # compute filters
#         sum_filt = torch.ones([1, 1, *win]).to("cuda")
#
#         pad_no = math.floor(win[0]/2)
#
#         if ndims == 1:
#             stride = (1)
#             padding = (pad_no)
#         elif ndims == 2:
#             stride = (1,1)
#             padding = (pad_no, pad_no)
#         else:
#             stride = (1,1,1)
#             padding = (pad_no, pad_no, pad_no)
#
#         # get convolution function
#         conv_fn = getattr(F, 'conv%dd' % ndims)
#
#         # compute CC squares
#         I2 = I * I
#         J2 = J * J
#         IJ = I * J
#
#         I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
#         J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
#         I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
#         J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
#         IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)
#
#         win_size = np.prod(win)
#         u_I = I_sum / win_size
#         u_J = J_sum / win_size
#
#         cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
#         I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
#         J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
#
#         cc = cross * cross / (I_var * J_var + 1e-5)
#
#         return -torch.mean(cc)