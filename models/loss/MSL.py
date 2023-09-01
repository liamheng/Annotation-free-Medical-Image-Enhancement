import torch

class MSL(torch.nn.Module):
    def __init__(self):
        super(MSL, self).__init__()

    def forward(self, x):
        """
        计算输入图像的像素点平方和
        :param x: 输入图像，大小为(batch_size, channel, height, width)
        :return: 像素点平方和
        """
        # 计算平方和
        sum_of_squares = torch.sum(torch.pow(x, 2))
        # rmse = torch.sqrt(sum_of_squares / torch.numel(x))
        rmse = sum_of_squares / torch.numel(x)
        return rmse
