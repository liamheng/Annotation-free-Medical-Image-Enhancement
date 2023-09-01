# -*- coding: UTF-8 -*-
"""
@Function:
@File: eval_lpips.py
@Date: 2021/4/14 22:03 
@Author: Hever
"""
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

import torch
img0 = torch.zeros(1,3,64,64) # images should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)
print(d)