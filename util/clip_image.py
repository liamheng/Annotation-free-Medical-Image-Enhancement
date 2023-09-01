import os
"""
   将图像一分为二
"""

from PIL import Image

image_dir = '../datasets/nidek_with_label'
save_dir = '../datasets/process_nidek/data'
save_mask_dir = '../datasets/process_nidek/mask'
def clip (image_dir,save_dir,mask_dir):

    # 获取当前文件夹所有文件名
    floder_name =[]
    for name in os.listdir(image_dir):
        path = image_dir + '/'+ name
        if isdir(path):
            floder_name.append(path)
    # 进入train 文件夹
    for path in floder_name:
        for name in os.listdir(path):
            # last_name = os.path.split(path)[-1]
            # pathid = last_name.split("_")[0]
            path_data = path + '/' + name
            if isdir(path_data):
                path_train = path_data + '/train'
                path_mask = path_train + '/mask'
                for name in os.listdir(path_train):
                    if name.endswith('bmp'):
                        image_path = path_train + '/' + name
                        last_name = os.path.split(image_path)[-1]
                        last_name = last_name.rsplit(".", 1)[0] # 去掉.bmp后缀
                        half(image_path , save_dir,last_name)
                for name in os.listdir(path_mask):
                    if name.endswith('bmp'):
                        image_path = path_mask + '/' + name
                        last_name = os.path.split(image_path)[-1]
                        last_name = last_name.rsplit(".", 1)[0] # 去掉.bmp后缀
                        half(image_path , mask_dir,last_name)
# 一分为二
def half(image_path,save_path,pathid):
    # 打开原始图片
    original_image = Image.open(image_path)
    # 获取原始图片的尺寸
    width, height = original_image.size

    # 计算新图片的尺寸
    new_width = width // 2
    new_height = height

    # 分割图片
    left_image = original_image.crop((0, 0, new_width, new_height))
    right_image = original_image.crop((new_width, 0, width, new_height))

    # 保存新图片
    left_image.save(os.path.join(save_path,pathid + "_1.png"))
    right_image.save(os.path.join(save_path,pathid + "_2.png"))

# 判断是否是一个文件夹
def isdir(path):
    return os.path.isdir(path)

if __name__ == '__main__':
    clip(image_dir,save_dir,save_mask_dir)

#  image_dir = '../datasets/nidek_with_label'
#
#  floder_name =[]
#  for name in os.listdir(image_dir):
#      path = image_dir + '/' + name
#      if isdir(path):
#       floder_name.append(path)
# pathid = []
# for path in floder_name:
#     last_name = os.path.split(path)[-1]
#     pathid.append(last_name.split("_")[0])




 # for path in floder_name:
 #     for name in os.listdir(path):
 #         path_data = path + '/' + name
 #         if isdir(path_data):
 #             path_train = path_data + '/train'
 #             path_mask = path_train + '/mask'
 #             for name in os.listdir(path_train):
 #                 if name.endswith('bmp'):
 #                     print(name)

