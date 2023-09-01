# -*- coding: UTF-8 -*-
"""
@Function:
@File: Unet_GM.py
@Date: 2021/7/25 16:32 
@Author: Hever
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.guided_filter_pytorch.HFC_filter import HFCFilter

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.hfc_filter = HFCFilter(filter_width, nsig)

    def forward(self, x):
        return self.double_conv(x)


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, is_leaky_relu=True, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if is_leaky_relu:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU()
            )
        # self.hfc_filter = HFCFilter(filter_width, nsig)

    def forward(self, x):
        return self.single_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downconv_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4,
                      stride=2, padding=1, bias=True),
            DoubleConv(in_channels, out_channels)
        )
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x, m):
        return self.downconv_conv(x), self.avg_pool(m)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = SingleConv(in_channels, out_channels, is_leaky_relu=False)

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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.conv(x))

class HFCConvOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFCConvOutConv, self).__init__()
        self.hfc = HFCFilter(23, 20)
        self.conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, d, u, m):
        # d_hfc = self.hfc(d, m)
        # u_hfc = self.hfc(u, m)
        # return self.tanh(self.conv(torch.cat([d, d_hfc, u, u_hfc], dim=1))), d_hfc, u_hfc

        # d_hfc = self.hfc(d, m)
        # u_hfc = self.hfc(u, m)
        return self.tanh(self.conv(torch.cat([d, u], dim=1))), d, u

class HFCConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFCConv, self).__init__()
        self.hfc = HFCFilter(23, 20)
        self.conv = nn.Conv2d(in_channels*4, out_channels, kernel_size=3, padding=1)
        # TODO:激活函数
        self.tanh = nn.Tanh()

    def forward(self, d, u, m):
        d_hfc = self.hfc(d, m)
        u_hfc = self.hfc(u, m)
        return self.conv(torch.cat([d, d_hfc, u, u_hfc], dim=1))

        # d_hfc = self.hfc(d, m)
        # u_hfc = self.hfc(u, m)
        # return self.tanh(self.conv(torch.cat([d, u], dim=1))), d, u

class UNet_GM(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet_GM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 256*256
        self.hfc_filter_x = HFCFilter(21, 20, sub_mask=True)
        # self.hfc_inc =
        self.inc = DoubleConv(self.n_channels, 64)
        # self.hfc_filter0 = HFCFilter(17, 20)

        # 128*128
        self.down1 = Down(64, 128)
        # self.hfc_filter1 = HFCFilter(13, 10)
        # 64*64
        self.down2 = Down(128, 256)
        # self.hfc_filter2 = HFCFilter(7, 7)
        # 32*32
        self.down3 = Down(256, 512)
        # self.hfc_filter3 = HFCFilter(5, 5)
        # 16*16
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # self.hfc_filter4 = HFCFilter(3, 3)
        # 32*32
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        # 64*64
        self.up2 = Up(512, 256 // factor, self.bilinear)
        # 128*128
        self.up3 = Up(256, 128 // factor, self.bilinear)
        # 256*256
        self.up4 = Up(128, 64, self.bilinear)
        # 256*256
        # self.hfc_conv = HFCConv(64, 64)
        # self.hfc_conv = HFCConv(64, 64)
        self.outc = OutConv(64, self.n_classes)
        print(self)

        # self.tanh = nn.Tanh()
        # self.conv_final = nn.Conv2d(6, 3, kernel_size=3, padding=1)


    def forward(self, x, mask):
        mask0 = mask
        # 256
        # x_hf = self.hfc_filter_x(x, mask0)
        # d0 = self.inc(torch.cat([x, x_hf], dim=1))
        d0 = self.inc(x)
        # h1 = self.hfc_filter0(x, mask0)
        # d0 = self.inc(x)
        # h1, mask1 = self.hfc_filter0(x, mask_x)
        # 128
        d1, mask1 = self.down1(d0, mask0)
        # 64
        d2, mask2 = self.down2(d1, mask1)
        # 32
        d3, mask3 = self.down3(d2, mask2)
        # 16
        b, _ = self.down4(d3, mask3)

        # 32
        u3 = self.up1(b, d3)
        # 64
        u2 = self.up2(u3, d2)
        # 128
        u1 = self.up3(u2, d1)
        # 256
        u0 = self.up4(u1, d0)
        # ch0 = self.hfc_conv(d0, u0, mask0)

        output_3 = self.outc(u0)
        # output_3_hf = self.hfc_filter_x(output_3, mask0)

        # final_output = self.conv_final(torch.cat([output_3, output_3_hf], dim=1))
        # return output_3, x_hf[:, :3], self.hfc_filter_x(x_hf, mask0)
        return output_3, output_3, output_3
        # logits, d_hfc0, u_hfc0 = self.hfc_conv_outc(output_3, x, mask0)
        #
        # return logits, d_hfc0, u_hfc0


if __name__ == '__main__':
    model = UNet_GM()
    model.to('cuda:0')
    input = torch.randn(1, 3, 256, 256).to('cuda:0')
    mask = torch.randn(1, 1, 256, 256).to('cuda:0')
    output = model(input, mask)
    print(output.shape)
    # model = UNet_GM()
