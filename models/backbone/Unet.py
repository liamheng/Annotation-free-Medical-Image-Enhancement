

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义双卷积层
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):

        if mid_channels is None:
            mid_channels = out_channels

        # 双卷积层由两个卷积层、两个批量归一化层和两个激活函数层组成
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) # inplace=True表示对输入的数据进行原地操作，节省内存
        )

# 定义下采样层
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

# 定义上采样层
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # align_corners=True表示对齐角点,scale_factor=2表示放大两倍,mode='bilinear'表示双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # in_channels // 2表示输入通道数的一半
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:#
        x1 = self.up(x1)
        # 如果高宽不是16的整数倍
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]  # 跳连接高差
        diff_x = x2.size()[3] - x1.size()[3]  # 跳连接宽差

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# 定义输出卷积层
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


# 定义U-Net网络
class Unet(nn.Module):
    def __init__(self,
                 input_nc: int = 3,
                 output_nc: int = 3,
                 segment_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(Unet, self).__init__()
        self.in_channels = input_nc
        self.num_classes = output_nc
        self.bilinear = bilinear

        self.in_conv = DoubleConv(input_nc, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1

        self.down4 = Down(base_c * 8, base_c * 16 // factor) # 这里没有进行通道数翻倍
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv1 = OutConv(base_c, output_nc)
        self.out_conv2 = OutConv(base_c, segment_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)

        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv1(x)
        segment = self.out_conv2(x)
        return logits,  segment


# 测试UNet
if __name__ == "__main__":
    net = Unet()
    x = torch.randn(8, 3, 480, 480)
    y1,y2 = net(x)
    print(y1.shape)
    print(y2.shape)
