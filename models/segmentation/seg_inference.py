# -*- coding: UTF-8 -*-
"""
@Function:
@File: eval_seg.py
@Date: 2022/1/14 9:51 
@Author: Hever
"""
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model_eval.segmentation.seg_cataract_inference_dataset import SegCataractInferenceDataset
from models.backbone.unet_open_source import UNet
import torchvision.transforms as transforms
import model_eval.segmentation.seg_metrics as segm
from util.util import tensor2im
from PIL import Image

# def te

transform_image = transforms.Compose(
        [transforms.Resize((256, 256), Image.BICUBIC), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
transform_mask = transforms.Compose(
    [transforms.Resize((256, 256), Image.BICUBIC), transforms.ToTensor()]
)

def tensor_to_numpy(image_tensor):
    image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    return image_numpy
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling


def eval_seg_init(pretrain_model, data_dir, mask_dir, gt_dir, gt_mask_dir, is_fake_B, is_fake_TB, device):
    inference_dataset = SegCataractInferenceDataset(data_dir, mask_dir, gt_dir=gt_dir,
                                            gt_mask_dir=gt_mask_dir, is_fake_B=is_fake_B, is_fake_TB=is_fake_TB)
    inference_loader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=4,
                                                   shuffle=False, num_workers=2)
    cudnn.benchmark = True
    model = UNet(3, 1)
    model.load_state_dict(torch.load('model_eval/segmentation/pretrain_model/{}'.format(pretrain_model),
                              map_location='cpu'))
    model.to(device)

    model.eval()
    return inference_loader, model

def eval_seg_model_init(pretrain_model, device):
    model = UNet(3, 1)
    model.load_state_dict(torch.load('model_eval/segmentation/pretrain_model/{}'.format(pretrain_model),
                              map_location='cpu'))
    model.to(device)
    model.eval()
    return model

def eval_seg_when_train(opt, test_web_dir, model):
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]))
    seg_dir = os.path.join(os.path.dirname(test_web_dir), '')
    if not os.path.isdir(seg_dir):
        os.mkdir(seg_dir)
    with torch.no_grad():
        torch.cuda.empty_cache()
        dataroot = opt.dataroot
        inference_dataset = SegCataractInferenceDataset(os.path.join(test_web_dir, 'images'), opt.seg_mask_dir,
                                                gt_dir=opt.seg_gt_dir,
                                                gt_mask_dir=opt.seg_gt_mask_dir, is_fake_B=True)
        inference_loader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=4,
                                                       shuffle=False, num_workers=2)
        metrics = eval_seg_run(inference_loader, model, os.path.join(dataroot, 'target_gt'), device, output_dir=seg_dir)
        return metrics


def eval_seg_run(model, data_dir, device, output_dir='', need_mask=False):
    # images = Image.open(image_path).convert('RGB')
    image_list = [image_name for image_name in os.listdir(data_dir) if image_name.find('png') >= 0 or image_name.find('jpg') >= 0]
    if not os.path.isdir(output_dir) and output_dir != '':
        os.mkdir(output_dir)
    with torch.no_grad():
        for epochID, (image_name) in enumerate(image_list):
            # 处理mask
            if image_name.find('mask')>=0:
                continue

            image_path = os.path.join(data_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image = transform_image(image).unsqueeze(0)
            image = image.to(device)
            ves_output = model(image)
            ves_output[ves_output >= 0.5] = 1
            ves_output[ves_output < 0.5] = 0
            if need_mask:
                mask_path = os.path.join(data_dir, 'mask.png')
                mask = Image.open(mask_path).convert('L')
                mask = transform_mask(mask).unsqueeze(0)
                mask = mask.to(device)
                ves_output = ves_output * mask

            if output_dir != '':
                for v, n in zip(ves_output, [image_name]):
                    im = tensor2gray_im(v)
                    save_path = os.path.join(output_dir, n)
                    save_image(im, save_path)



def tensor2gray_im(im):
    image_tensor = im.data
    image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    # image_numpy = np.tile(image_numpy, (1, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255
    return image_numpy

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(np.uint8(image_numpy))
    # h, w = image_numpy.shape
    image_pil.save(image_path.replace('jpg', 'png'))

if __name__ == '__main__':

    # Setting parameters
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--model_dir', type=str, default='./pretrain_model')
    parser.add_argument('--image_dir', type=str, default=r'./images/input')
    parser.add_argument('--output_dir', type=str, default=r'./images/output')

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--need_mask', action='store_true')


    args = parser.parse_args()
    print(args)
    transform_image = transforms.Compose(
        [transforms.Resize((256, 256), Image.BICUBIC), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )


    # TODO: 做一个TestDataset，把gt也load进来，直接评价，并且保存

    cudnn.benchmark = True
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_name_list = os.listdir(image_dir)
    seg_model = 'seg_s3.pth'
    cudnn.benchmark = True
    model = UNet(3, 1)
    model.load_state_dict(torch.load('model_eval/segmentation/pretrain_model/{}'.format(seg_model),
                                     map_location='cpu'))
    model.to(device)

    model.eval()
    for image_name in image_name_list:

        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(image_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        image = transform_image(image)
        image = image.unsqueeze(dim=0).to(device)
        res = model(image)
        save_image(tensor2gray_im(res), os.path.join(output_dir, image_name.split('.')[0]+'.png'))


        # torch.cuda.empty_cache()



