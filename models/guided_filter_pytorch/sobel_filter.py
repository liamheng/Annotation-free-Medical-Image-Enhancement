from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
# import kornia
import torch
import torchvision.transforms as transforms
from .box_filter import BoxFilter

class ThreeSobelFilter(nn.Module):
    def __init__(self, device):
        super(ThreeSobelFilter, self).__init__()
        self.sobel_filter_3 = SobelFilter(device, 3)
        self.sobel_filter_5 = SobelFilter(device, 5)
        self.sobel_filter_7 = SobelFilter(device, 7)

    def forward(self, x):
        sobel_3 = self.sobel_filter_3(x)
        sobel_5 = self.sobel_filter_5(x)
        sobel_7 = self.sobel_filter_7(x)
        sobel_x = torch.cat([sobel_3, sobel_5, sobel_7], dim=1)
        return sobel_x


class OneSobelFilter(nn.Module):
    def __init__(self, device):
        super(OneSobelFilter, self).__init__()
        self.sobel_filter_3 = SobelFilter(device, 3)

    def forward(self, x):
        sobel_3 = self.sobel_filter_3(x)
        return sobel_3

class SobelFilter(nn.Module):
    def __init__(self, device, kernel_size=3):
        super(SobelFilter, self).__init__()
        self.device = device
        if kernel_size == 7:
            self.X = torch.tensor([[1, 6, 15, 20, 15, 6, 1]]).float()
            self.W = torch.tensor([[-1, -2, -3, 0, 3, 2, 1]]).float()
            self.padding = nn.ReplicationPad2d(3)
        elif kernel_size == 5:
            self.X = torch.tensor([[1, 4, 6, 4, 1]]).float()
            self.W = torch.tensor([[-1, -2, 0, 2, 1]]).float()
            self.padding = nn.ReplicationPad2d(2)
        else:
            self.X = torch.tensor([[1, 2, 1]]).float()
            self.W = torch.tensor([[-1, 0, 1]]).float()
            self.padding = nn.ReplicationPad2d(1)

        self.kernel_x = torch.mm(self.W.T, self.X)
        # 归一化
        # self.kernel_x /= (self.kernel_x.abs().sum() / 2)
        self.kernel_x = self.kernel_x.reshape([1, 1, kernel_size, kernel_size])
        self.kernel_y = self.kernel_x.transpose(2, 3)

        self.kernel_x = self.kernel_x.to(self.device)
        self.kernel_y = self.kernel_y.to(self.device)
        self.kernel_x.requires_grad = False
        self.kernel_y.requires_grad = False

    def forward(self, x):
        # x = x + 1
        x = self.padding(x)

        sobel_x = torch.abs(F.conv2d(x, self.kernel_x))
        sobel_y = torch.abs(F.conv2d(x, self.kernel_y))
        sobel_im = (sobel_x + sobel_y) / 2

        # 归一化
        # sobel_im = sobel_im - 1
        sobel_im = ((sobel_im / sobel_im.max()) - 0.5) / 0.5
        return sobel_im

class FourSobelFilter(nn.Module):
    def __init__(self, device):
        super(FourSobelFilter, self).__init__()
        self.sobel_filter_3 = SobelFilter_diff_directions(device, 3)

    def forward(self, x):
        sobel_3 = self.sobel_filter_3(x)
        return sobel_3

class SobelFilter_diff_directions(nn.Module):
    def __init__(self, device, kernel_size=3):
        super(SobelFilter_diff_directions, self).__init__()
        self.device = device

        # self.X = torch.tensor([[1, 2, 1]]).float()
        # self.W = torch.tensor([[-1, 0, 1]]).float()
        # self.padding = nn.ReplicationPad2d(1)
        #
        # self.kernel_x = torch.mm(self.W.T, self.X)
        # # 归一化
        # # self.kernel_x /= (self.kernel_x.abs().sum() / 2)
        # self.kernel_x = self.kernel_x.reshape([1, 1, kernel_size, kernel_size])
        # self.kernel_y = self.kernel_x.transpose(2, 3)
        # 4个方向的滤波器
        self.kernel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float()
        self.kernel_y = torch.tensor([[2,1,0],[1,0,-1],[0,-1,-2]]).float()
        self.kernel_w = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float()
        self.kernel_z = torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]]).float()

        #  归一化
        self.kernel_x /= (self.kernel_x.abs().sum() / 2)
        self.kernel_y /= (self.kernel_y.abs().sum() / 2)
        self.kernel_w /= (self.kernel_w.abs().sum() / 2)
        self.kernel_z /= (self.kernel_z.abs().sum() / 2)
        self.kernel_x = self.kernel_x.reshape([1, 1, kernel_size, kernel_size])
        self.kernel_y = self.kernel_y.reshape([1, 1, kernel_size, kernel_size])
        self.kernel_w = self.kernel_w.reshape([1, 1, kernel_size, kernel_size])
        self.kernel_z = self.kernel_z.reshape([1, 1, kernel_size, kernel_size])

        self.padding = nn.ReplicationPad2d(1)
        self.kernel_x = self.kernel_x.to(self.device)
        self.kernel_y = self.kernel_y.to(self.device)
        self.kernel_w = self.kernel_w.to(self.device)
        self.kernel_z = self.kernel_z.to(self.device)

        self.kernel_x.requires_grad = False
        self.kernel_y.requires_grad = False
        self.kernel_w.requires_grad = False
        self.kernel_z.requires_grad = False

    def forward(self, x):
        # x = x + 1
        x = self.padding(x)

        # sobel_x = torch.abs(F.conv2d(x, self.kernel_x))
        # sobel_y = torch.abs(F.conv2d(x, self.kernel_y))
        # sobel_w = torch.abs(F.conv2d(x, self.kernel_w))
        # sobel_z = torch.abs(F.conv2d(x, self.kernel_z))
        sobel_x = torch.sqrt(torch.pow(F.conv2d(x, self.kernel_x), 2))
        sobel_y = torch.sqrt(torch.pow(F.conv2d(x, self.kernel_y), 2))
        sobel_w = torch.sqrt(torch.pow(F.conv2d(x, self.kernel_w), 2))
        sobel_z = torch.sqrt(torch.pow(F.conv2d(x, self.kernel_z), 2))
        sobel_im = (sobel_x + sobel_y +  sobel_w + sobel_z) / 4

        # 归一化
        # sobel_im = sobel_im - 1
        sobel_im = ((sobel_im / sobel_im.max()) - 0.5) / 0.5
        return sobel_im
