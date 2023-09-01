import os
import glob
from multiprocessing.pool import Pool

import numpy as np
from utils_de import imread, imwrite
from PIL import Image
from degrad_de import *
from skimage import util
import random
import json

sizeX = 512
sizeY = 512


type_list = [
    '00100', '10000', '10100', '00001',
    '00101', '10001', '10101', '00010',
    '00110', '10010', '10110']
# de_image_type_list = [
#     '00',
#     '10',
#     '01'
# ]
# '111' means: DE_BLUR, DE_SPOT, DE_ILLUMINATION
# '1000' means: salt and pepper

def process(image_list, output_image_dir):

    for image_path in image_list: 
        name_ext = image_path.split('/')[-1]
        name = name_ext.split('\\')[-1].split('.')[0]
        
        img = Image.open(image_path).convert('L')
        img = img.resize((sizeX, sizeY), Image.BICUBIC)
        mask = np.ones(img.size)
        # mask = Image.open(mask_path).convert('L')
        # mask = mask.resize((sizeX, sizeY), Image.BICUBIC)
        # mask = np.expand_dims(mask, axis=2)
        # mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
        for i in range(16):
        # for t in
        # :
            t = random.sample(type_list, 1)[0]
            de_type = t[:3]
            de_image_type = t[3:]
            r_img, r_params = DE_process(img, mask, sizeX, sizeY, de_type, de_image_type)
            dst_img = os.path.join(output_image_dir, name+'-'+str(i)+'.png')
            imwrite(dst_img, r_img)
            # param_dict = {'name':name_ext, 'type':t, 'params':r_params}
            # with open(os.path.join('./images/de_js_file', name+'_'+t+'.json'), 'w') as json_file:
            #     json.dump(param_dict, json_file)

        
if __name__=="__main__":
    # original_image_dir = r'D:\dataset\RAVIR Dataset\noise_data\original_image'
    # output_image_dir = r'D:\dataset\RAVIR Dataset\noise_data\noise_image'
    original_image_dir = '../../../datasets/process_nidek/data'
    output_image_dir = '../../../datasets/process_nidek/noisy_data'
    # 函数返回一个匹配指定模式的路径名列表。例如，要查找当前工作目录下所有扩展名为 .png 的文
    image_list = glob.glob(os.path.join(original_image_dir, '*.png'))  # 读取所有图片
    patches = 16
    patch_len = int(len(image_list)/patches)
    process(image_list, output_image_dir)
    # filesPatchList = []
    # for i in range(patches-1):
    #     fileList = image_list[i*patch_len:(i+1)*patch_len]
    #     filesPatchList.append(fileList)
    # filesPatchList.append(image_list[(patches-1)*patch_len:])
    #
    # # mutiple process
    # pool = Pool(patches)
    # pool.map(process, filesPatchList)
    # pool.close()
