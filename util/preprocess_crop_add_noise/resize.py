# -*- coding: UTF-8 -*-
"""
@Function:
@File: resize.py
@Date: 2023/1/10 17:09 
@Author: Hever
"""
import cv2
import os

# crop_width = 10
# image_input_dir = './50'
# image_input_dir = r'D:\dataset\RAVIR Dataset\noise_data\original_image'
# image_output_dir = 'resize_512'
# image_output_dir = r'D:\dataset\RAVIR Dataset\noise_data\resized_image'
image_input_dir = r'RGBIR_test'
image_output_dir = r'RGBIR_test_resize'
for dirpath, dirnames, filenames in os.walk(image_input_dir):
    for filename in filenames:
        if 'bmp' in filename:
            image = cv2.imread(os.path.join(dirpath, filename), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
            if not os.path.isdir(dirpath.replace(image_input_dir, image_output_dir)):
                os.makedirs(dirpath.replace(image_input_dir, image_output_dir))
            cv2.imwrite(os.path.join(dirpath.replace(image_input_dir, image_output_dir), filename.replace('bmp', 'png')), image)

# image_list = os.listdir(image_input_dir)
# for image_name in image_list:
#     images = cv2.imread(os.path.join(image_input_dir, image_name), cv2.IMREAD_GRAYSCALE)
#     images = cv2.resize(images, (512, 512), interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(os.path.join(image_output_dir, image_name), images)


