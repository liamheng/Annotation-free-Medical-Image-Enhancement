# -*- coding: UTF-8 -*-
"""
@Function:
@File: resize.py
@Date: 2023/1/10 17:09 
@Author: Hever
"""
import cv2
import os



size = 316
image_input_dir = './resize_512'
image_output_dir = './resize_and_crop'
image_list = os.listdir(image_input_dir)
for image_name in image_list:
    if 'R_' in image_name:
        crop_y = 80
        crop_x = 512-size
        image = cv2.imread(os.path.join(image_input_dir, image_name), cv2.IMREAD_GRAYSCALE)
        crop_image = image[crop_y:crop_y+size, crop_x:crop_x+size]
        # images = cv2.resize(images, (512, 512))
        cv2.imwrite(os.path.join(image_output_dir, image_name), crop_image)
    else:
        crop_y = 80
        crop_x = 0
        image = cv2.imread(os.path.join(image_input_dir, image_name), cv2.IMREAD_GRAYSCALE)
        crop_image = image[crop_y:crop_y + size, crop_x:crop_x + size]
        # images = cv2.resize(images, (512, 512))
        cv2.imwrite(os.path.join(image_output_dir, image_name), crop_image)


