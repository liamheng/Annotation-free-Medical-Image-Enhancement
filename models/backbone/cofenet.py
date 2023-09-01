# -*- coding: UTF-8 -*-
"""
@Function:
@File: cofenet.py
@Date: 2022/1/4 17:25 
@Author: Hever
"""
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from torchvision import models
from functools import partial
import torch.nn.functional as F


nonlinearity = partial(F.relu, inplace=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, in_kernel, in_pad, in_bias):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
        self.relu1 = nonlinearity
        self.conv2 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
        self.relu2 = nonlinearity

    def forward(self, x):
        x0 = self.conv1(x)
        x = self.relu2(x0)
        x = self.conv2(x)
        x = x0 + x
        out = self.relu2(x)
        return out


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x



#  concat
class CofeNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(CofeNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
        self.G_relu4 = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)

        self.G_pool_256 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_64 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_input_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=use_bias)

        #### CE_Net #####
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

        #### begin attention module ####
        self.a_in_pool = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=use_bias)
        self.a_en_relu1 = nn.ReLU()
        self.a_en_pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=use_bias)
        self.a_en_relu2 = nn.ReLU()
        self.a_en_pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=use_bias)
        self.a_en_relu3 = nn.ReLU()
        self.a_en_pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_de_conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=use_bias)
        self.a_de_relu31 = nn.ReLU()
        self.a_de_deconv3 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=use_bias)
        self.a_de_relu32 = nn.ReLU()

        self.a_de_conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=use_bias)
        self.a_de_relu21 = nn.ReLU()
        self.a_de_deconv2 = nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=use_bias)
        self.a_de_relu22 = nn.ReLU()

        self.a_de_conv1 = nn.Conv2d(128, 64, 3, padding=1, bias=use_bias)
        self.a_de_relu11 = nn.ReLU()
        self.a_de_deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)
        self.a_de_relu12 = nn.ReLU()

        self.a_spot_256 = nn.Conv2d(64, 1, 1, padding=0, bias=use_bias)
        self.a_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

    def forward(self, input_512, input_256, input_norm):

        ### begin segmentation network ###
        x = self.firstconv(input_norm)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = F.sigmoid(out)

        ###　end segmentation network ###
        ### begin attention network ###
        x = self.a_in_pool(input_512)
        x = self.a_en_conv1(x)
        x = self.a_en_relu1(x)
        x = self.a_en_pool1(x)

        x = self.a_en_conv2(x)
        x = self.a_en_relu2(x)
        x = self.a_en_pool2(x)

        x = self.a_en_conv3(x)
        x = self.a_en_relu3(x)
        x = self.a_en_pool3(x)

        x = self.a_de_conv3(x)
        x = self.a_de_relu31(x)
        x = self.a_de_deconv3(x)
        x = self.a_de_relu32(x)

        x = self.a_de_conv2(x)
        x = self.a_de_relu21(x)
        x = self.a_de_deconv2(x)
        x = self.a_de_relu22(x)

        x = self.a_de_conv1(x)
        x = self.a_de_relu11(x)
        x = self.a_de_deconv1(x)
        x = self.a_de_relu12(x)

        a_mask = self.a_spot_256(x)

        ### end attention network ###
        ### begin enhance network ###
        d1_1 = self.G_pool_256(d1)
        d2_1 = self.G_pool_128(d2)
        d3_1 = self.G_pool_64(d3)
        a_mask_1 = self.a_pool_128(a_mask)

        input_copy_256 = torch.cat([input_256, input_256], 1)
        x = self.G_conv1(input_copy_256)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        x = x * a_mask_1.expand_as(x) + x
        con_2 = torch.cat([x, d1_1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2_1], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3_1], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_256 = F.tanh(x)
        input_2 = self.G_input_2(output_256)

        # ori_scale
        input_copy_512 = torch.cat([input_512, input_2], 1)
        x = self.G_conv1(input_copy_512)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        x = x * a_mask.expand_as(x) + x
        con_2 = torch.cat([x, d1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_512 = F.tanh(x)

        ##＃end enhancement network ###

        # output enhancement results
        # out segmentation result
        # fusion feature d3 d2 d1
        # d3 1/8 feature 128 channel
        # d2 1/4 feature 64 channel
        # d1 1/2 feature 64 channel
        return output_512, output_256, out, a_mask# class CofeNet(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
#         super(CofeNet, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         self.G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
#         self.G_relu1 = nn.ReLU()
#         self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
#
#         self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
#         self.G_relu2_0 = nn.ReLU()
#         # concat 1/2
#         self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
#         self.G_relu2 = nn.ReLU()
#         self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#
#         self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
#         self.G_relu3_0 = nn.ReLU()
#         # concat 1/4
#         self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
#         self.G_relu3 = nn.ReLU()
#         self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#
#         self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
#         self.G_relu4_0 = nn.ReLU()
#         # concat 1/8
#         self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
#         self.G_relu4 = nn.ReLU()
#         self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
#
#         self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)
#
#         self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)
#
#         self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)
#
#         self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
#         self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)
#
#         self.G_pool_256 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
#         self.G_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
#         self.G_pool_64 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
#         self.G_input_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=use_bias)
#         #### CE_Net ######
#
#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.dblock = DACblock(512)
#         self.spp = SPPblock(512)
#
#         self.decoder4 = DecoderBlock(516, filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)
#
#     def forward(self, input_512, input_256, input_norm):
#
#         ### begin segmentation network ###
#         x = self.firstconv(input_norm)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#
#         # Center
#         e4 = self.dblock(e4)
#         e4 = self.spp(e4)
#
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         out = F.sigmoid(out)
#
#         ###　end segmentation network ###
#
#         ### begin enhance network ###
#         d1_1 = self.G_pool_256(d1)
#         d2_1 = self.G_pool_128(d2)
#         d3_1 = self.G_pool_64(d3)
#
#         input_copy_256 = torch.cat([input_256, input_256], 1)
#         x = self.G_conv1(input_copy_256)
#         x = self.G_relu1(x)
#         x = self.G_conv1_1(x)
#         x = self.G_conv1_2(x)
#         x_512 = self.G_conv1_3(x)
#
#         x = self.G_conv2_0(x_512)
#         x = self.G_relu2_0(x)
#         con_2 = torch.cat([x, d1_1], 1)
#         x = self.G_conv2(con_2)
#         x = self.G_relu2(x)
#         x = self.G_conv2_1(x)
#         x = self.G_conv2_2(x)
#         x_256 = self.G_conv2_３(x)
#
#         x = self.G_conv3_0(x_256)
#         x = self.G_relu3_0(x)
#         con_4 = torch.cat([x, d2_1], 1)
#         x = self.G_conv3(con_4)
#         x = self.G_relu3(x)
#         x = self.G_conv3_1(x)
#         x = self.G_conv3_2(x)
#         x_128 = self.G_conv3_3(x)
#
#         x = self.G_conv4_0(x_128)
#         x = self.G_relu4_0(x)
#         con_8 = torch.cat([x, d3_1], 1)
#         x = self.G_conv4(con_8)
#         x = self.G_relu4(x)
#         x = self.G_conv4_1(x)
#         x = self.G_conv4_2(x)
#         x = self.G_conv4_3(x)
#
#         x = self.G_deconv4_3(x)
#         x = self.G_deconv4_2(x)
#         x = self.G_deconv4_1(x)
#         x = self.G_deconv4_0(x)
#
#         x = x + x_128
#
#         x = self.G_deconv3_3(x)
#         x = self.G_deconv3_2(x)
#         x = self.G_deconv3_1(x)
#         x = self.G_deconv3_0(x)
#
#         x = x + x_256
#
#         x = self.G_deconv2_3(x)
#         x = self.G_deconv2_2(x)
#         x = self.G_deconv2_1(x)
#         x = self.G_deconv2_0(x)
#
#         x = x + x_512
#
#         x = self.G_deconv1_3(x)
#         x = self.G_deconv1_2(x)
#         x = self.G_deconv1_1(x)
#         x = self.G_deconv1_0(x)
#         output_256 = F.sigmoid(x)
#         input_2 = self.G_input_2(output_256)
#
#         # ori_scale
#         input_copy_512 = torch.cat([input_512, input_2], 1)
#         x = self.G_conv1(input_copy_512)
#         x = self.G_relu1(x)
#         x = self.G_conv1_1(x)
#         x = self.G_conv1_2(x)
#         x_512 = self.G_conv1_3(x)
#
#         x = self.G_conv2_0(x_512)
#         x = self.G_relu2_0(x)
#         con_2 = torch.cat([x, d1], 1)
#         x = self.G_conv2(con_2)
#         x = self.G_relu2(x)
#         x = self.G_conv2_1(x)
#         x = self.G_conv2_2(x)
#         x_256 = self.G_conv2_３(x)
#
#         x = self.G_conv3_0(x_256)
#         x = self.G_relu3_0(x)
#         con_4 = torch.cat([x, d2], 1)
#         x = self.G_conv3(con_4)
#         x = self.G_relu3(x)
#         x = self.G_conv3_1(x)
#         x = self.G_conv3_2(x)
#         x_128 = self.G_conv3_3(x)
#
#         x = self.G_conv4_0(x_128)
#         x = self.G_relu4_0(x)
#         con_8 = torch.cat([x, d3], 1)
#         x = self.G_conv4(con_8)
#         x = self.G_relu4(x)
#         x = self.G_conv4_1(x)
#         x = self.G_conv4_2(x)
#         x = self.G_conv4_3(x)
#
#         x = self.G_deconv4_3(x)
#         x = self.G_deconv4_2(x)
#         x = self.G_deconv4_1(x)
#         x = self.G_deconv4_0(x)
#
#         x = x + x_128
#
#         x = self.G_deconv3_3(x)
#         x = self.G_deconv3_2(x)
#         x = self.G_deconv3_1(x)
#         x = self.G_deconv3_0(x)
#
#         x = x + x_256
#
#         x = self.G_deconv2_3(x)
#         x = self.G_deconv2_2(x)
#         x = self.G_deconv2_1(x)
#         x = self.G_deconv2_0(x)
#
#         x = x + x_512
#
#         x = self.G_deconv1_3(x)
#         x = self.G_deconv1_2(x)
#         x = self.G_deconv1_1(x)
#         x = self.G_deconv1_0(x)
#         output_512 = F.sigmoid(x)
#
#         return output_512, output_256, out
