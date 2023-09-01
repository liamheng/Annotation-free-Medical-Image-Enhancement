# 计算信息熵
import torch


class entropy(torch.nn.Module):

    def __init__(self):
        super(entropy, self).__init__()

    def forward(self, x):
        # proba = x.softmax(1)  # softmax(1) 求每个类别的softmax
        n, c, h, w = x.size()

        # entropy_loss    \表示换行
        # entropy = -(proba * torch.log2(proba + 1e-10)).sum() / \
        #           (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
        # entropy = -(x * torch.log2(x + 1e-10)).sum() / (n * h * w) # 只会分割出mask
        entropy = -(x * torch.log2(x + 1e-10)+(1-x)* torch.log2(1-x + 1e-10)).sum()/(n * h * w )


        return entropy