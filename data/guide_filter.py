import numpy as np
import cv2
import os

def guideFilter(I, p, winSize, eps):
    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑

    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差

    a = cov_Ip / (var_I + eps)         # 相关因子a
    # b = mean_p - a * mean_I            # 相关因子b

    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑

    return mean_a


def crop(img):
    h, w, c = img.shape
    return img[:, 0:int(w/2), :], img[:, int(w/2):, :]