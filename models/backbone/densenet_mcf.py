# encoding: utf-8
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F


class DenseNet121_v0(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, n_class):
        super(DenseNet121_v0, self).__init__()
        # rue表示不用重新下载，false会自动下载模型（需要翻墙）
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        # 在DenseNet121的基础上，修改了全连接层(注意是修改)，并且增加了个sigmoid函数
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, n_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入通过densenet121处理的到输出结果
        x = self.densenet121(x)
        return x


# A_model = DenseNet121_v0(n_class=3)
# print(torchvision.models.densenet121(pretrained=False).features)
class dense121_mcs(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, n_class):
        super(dense121_mcs, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=False)  # 拿到densenet的实例化
        num_ftrs = self.densenet121.classifier.in_features  # 全连接层的入参

        A_model = DenseNet121_v0(n_class=n_class)
        self.featureA = A_model  # 修改了的densenet121的实例化（创立三次实例化，是需要三次独立修改）
        self.classA = A_model.densenet121.features  # 除了全连接层的其他层

        B_model = DenseNet121_v0(n_class=n_class)
        self.featureB = B_model  # 修改了的densenet121的实例化
        self.classB = B_model.densenet121.features  #

        C_model = DenseNet121_v0(n_class=n_class)
        self.featureC = C_model  # 修改了的densenet121的实例化
        self.classC = C_model.densenet121.features

        self.combine1 = nn.Sequential(  # Prediction-level  四个输出做全连接
            nn.Linear(n_class * 4, n_class),
            nn.Sigmoid()
        )

        self.combine2 = nn.Sequential(  # Feature-level
            nn.Linear(num_ftrs * 3, n_class),
            nn.Sigmoid()
        )

    def forward(self, x, y, z):


        x1 = self.featureA(x)  # rgb torch.Size([4, 2])  (batch-size,质量分类数)  输出层
        y1 = self.featureB(y)  # lab
        z1 = self.featureC(z)  # hsv
        x2 = self.classA(x)  # rgb - 排除classifier torch.Size([4, 1024, 7, 7]) （batch size, channel, height, width）

        x2 = F.relu(x2, inplace=True)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)  # 对in_features进行下采样，并且拉平 torch.Size([4, 1024]) 全连接

        y2 = self.classB(y)  # lab - 排除classifier
        y2 = F.relu(y2, inplace=True)
        y2 = F.adaptive_avg_pool2d(y2, (1, 1)).view(y2.size(0), -1)
        z2 = self.classC(z)  # hsv - 排除classifier
        z2 = F.relu(z2, inplace=True)
        z2 = F.adaptive_avg_pool2d(z2, (1, 1)).view(z2.size(0), -1)

        # Feature - Level
        combine = torch.cat((x2.view(x2.size(0), -1),
                             y2.view(y2.size(0), -1),
                             z2.view(z2.size(0), -1)), 1)  # 按列拼接 torch.Size([4, 3072]) Feature-level

        # combine2(3072，2) 全连接层 , 三钟特征融合之后做分类
        combine = self.combine2(combine)  # torch.Size([4, 2])

        # Prediction - level
        combine3 = torch.cat((x1.view(x1.size(0), -1),
                              y1.view(y1.size(0), -1),
                              z1.view(z1.size(0), -1),
                              combine.view(combine.size(0), -1)), 1)  # torch.Size([4, 8])

        #  四个分类级做融合后，再做一次分类
        combine3 = self.combine1(combine3)  # torch.Size([4, 2])

        return x1, y1, z1, combine, combine3  # out_A, out_B, out_C, out_F, combine
