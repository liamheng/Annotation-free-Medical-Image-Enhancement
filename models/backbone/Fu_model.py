# -*- coding: UTF-8 -*-
"""
@Function:
@File: LQA_Unet.py
@Date: 2021/7/14 16:45 
@Author: Hever
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.RSA_module import RSAModule
import numpy as np
from models.networks import init_weights

class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.triple_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            TripleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, padding=1, kernel_size=4, stride=2)
        self.conv = TripleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvSigmoid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvSigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class OutConvTanh(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvTanh, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.conv(x))

class LQAModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(LQAModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = TripleConv(self.in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.bottom = TripleConv(512, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.outc = OutConvSigmoid(64, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # b = self.bottom(x4)
        u3 = self.up3(x4, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up1(u2, x1)
        logits = self.outc(u1)

        return logits


class CorrectionModule(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(CorrectionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = TripleConv(self.in_channels, 64)
        self.down_pool1 = nn.MaxPool2d(2)
        self.reduction1 = nn.Conv2d(64 + 64 + 64, 64, kernel_size=3, padding=1)
        self.down_conv1 = TripleConv(64, 128)
        self.down_pool2 = nn.MaxPool2d(2)
        self.reduction2 = nn.Conv2d(128 * 2, 128, kernel_size=3, padding=1)
        self.down_conv2 = TripleConv(128, 256)
        self.down_pool3 = nn.MaxPool2d(2)
        self.reduction3 = nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1)
        self.down_conv3 = TripleConv(256, 512)
        # self.bottom = TripleConv(512, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.outc = OutConvTanh(64, self.out_channels)

    def forward(self, x, artifact_mask, R_u1, R_u2, R_u3):
        x1 = self.inc(x)

        x2 = self.down_pool1(x1)
        artifact_mask_feature = artifact_mask * x2 + x2
        x2 = self.reduction1(torch.cat([x2, artifact_mask_feature, R_u1], dim=1))
        x2 = self.down_conv1(x2)

        x3 = self.down_pool2(x2)
        x3 = self.reduction2(torch.cat([x3, R_u2], dim=1))
        x3 = self.down_conv2(x3)

        x4 = self.down_pool3(x3)
        x4 = self.reduction3(torch.cat([x4, R_u3], dim=1))
        x4 = self.down_conv3(x4)

        u3 = self.up3(x4, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up1(u2, x1)
        logits = self.outc(u1)

        return logits


class Fu_model(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super(Fu_model, self).__init__()
        self.RSA = RSAModule(in_channels=3, out_channels=1)
        self.correction = CorrectionModule(in_channels=6, out_channels=3)
        self.LQA = LQAModule(in_channels=3, out_channels=1)
        self.device = 'cuda:{}'.format(gpu_ids[0])
        self.init_net(gpu_ids, init_type, init_gain)

    def init_net(self, gpu_ids, init_type, init_gain):
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.to(self.device)
        init_weights(self.LQA, init_type, init_gain)
        init_weights(self.correction, init_type, init_gain)
        # load已经训练好的RSA到GPU并且不需要梯度
        self.RSA.load_state_dict(torch.load('./pre_trained_model/net_RSA.pth', map_location=str(self.device)))
        self.set_requires_grad(self.RSA, requires_grad=False)
        print()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, x1, x2):
        R_u1, R_u2, R_u3, seg_mask = self.RSA(x1)
        x1_pool = F.interpolate(x1, scale_factor=1/2, mode='bilinear', align_corners=True)
        artifact_mask = self.LQA(x1_pool)
        c_input = torch.cat([x1, x2], dim=1)
        final_output = self.correction(c_input, artifact_mask,
                                       R_u1, R_u2, R_u3)
        artifact_mask_resize = F.interpolate(artifact_mask, scale_factor=2, mode='bilinear', align_corners=True)
        return artifact_mask_resize, seg_mask, final_output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--init_gain', type=float, default=0.02)
    args = parser.parse_args()
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    input1 = torch.FloatTensor(np.random.random([2, 3, 256, 256])).to('cuda')
    # input2 = F.avg_pool2d(input1, kernel_size=(2,2), stride=2)
    input2 = F.interpolate(input1, scale_factor=1/2, mode='bilinear', align_corners=True)
    input3 = F.interpolate(input2, scale_factor=2, mode='bilinear', align_corners=True)
    input4 = F.interpolate(input3, scale_factor=1/2, mode='bilinear', align_corners=True)
    # input4 = F.avg_pool2d(input3, kernel_size=(2,2), stride=2)


    network = Fu_model(3, 3, gpu_ids=args.gpu_ids)

    output_layer1 = network(input1, input3)
    output_layer2 = network(input2, input4)

    print()