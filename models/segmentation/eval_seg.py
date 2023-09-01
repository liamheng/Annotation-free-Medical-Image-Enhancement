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
from model_eval.segmentation.seg_inference_dataset import SegInferenceDataset
from models.backbone.unet_open_source import UNet
import model_eval.segmentation.seg_metrics as segm
from util.util import tensor2im
from PIL import Image

# def te
def tensor_to_numpy(image_tensor):
    image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    return image_numpy
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling


def eval_seg_init(pretrain_model, data_dir, mask_dir, gt_dir, gt_mask_dir, is_fake_B, is_fake_TB, device):
    inference_dataset = SegInferenceDataset(data_dir, mask_dir, gt_dir=gt_dir,
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

def eval_seg_when_train(opt, test_web_dir, model, is_fake_B=True, is_fake_TB=False):
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]))
    seg_dir = os.path.join(test_web_dir, 'segmentation')
    if not os.path.isdir(seg_dir):
        os.mkdir(seg_dir)
    with torch.no_grad():
        torch.cuda.empty_cache()
        dataroot = opt.dataroot
        inference_dataset = SegInferenceDataset(os.path.join(test_web_dir, 'images'), opt.seg_mask_dir,
                                                gt_dir=opt.seg_gt_dir,
                                                gt_mask_dir=opt.seg_gt_mask_dir, is_fake_B=is_fake_B, is_fake_TB=is_fake_TB)
        inference_loader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=4,
                                                       shuffle=False, num_workers=2)
        metrics = eval_seg_run(inference_loader, model, os.path.join(dataroot, 'target_gt'), device, output_dir=seg_dir)
        return metrics


def eval_seg_run(dataloader, model, gt_dir, device, output_dir=''):
    if not os.path.isdir(output_dir) and output_dir != '':
        os.mkdir(output_dir)
    gt_im_list = []
    ves_output_im_list = []
    valid_mask_im_list = []
    print(len(dataloader))
    with torch.no_grad():
        for epochID, (input_data) in enumerate(dataloader):
            image = input_data['images'].to(device)
            image_name = input_data['image_name']
            ves_output = model(image)
            ves_output[ves_output >= 0.5] = 1
            ves_output[ves_output < 0.5] = 0
            if gt_dir != '':
                gt = input_data['gt'].to(device)
                mask = input_data['mask_union'].to(device)

                ves_output_im_list.append(tensor_to_numpy(ves_output))
                gt_im_list.append(tensor_to_numpy(gt))
                valid_mask_im_list.append(tensor_to_numpy(mask))
            if output_dir != '':
                for v, n in zip(ves_output, image_name):
                    im = tensor2gray_im(v)
                    save_path = os.path.join(output_dir, n)
                    save_image(im, save_path)
            # if epochID % 10 == 0:
            #     print(epochID)
        if gt_dir != '':
            gt_ims_np = np.concatenate(gt_im_list, axis=0)
            ves_output_ims_np = np.concatenate(ves_output_im_list, axis=0)
            valid_mask_ims_np = np.concatenate(valid_mask_im_list, axis=0)

            tpfptnfn = segm.tpfptnfn(gt_ims_np, ves_output_ims_np, valid_mask_ims_np)
            metrics = segm.segmentation_metrics(tpfptnfn, decimals=3)
        else:
            return None
        return metrics


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
    parser.add_argument('--data_root', type=str, default='/images/liuhaofeng/Project/pixDRP/results')
    parser.add_argument('--data_dir', type=str, default='/images/liuhaofeng/Project/pixDRP/datasets/fiq_1229/target_image_multi_seg')
    parser.add_argument('--mask_dir', type=str, default='/images/liuhaofeng/Project/pixDRP/datasets/fiq_1229/target_mask')
    parser.add_argument('--gt_dir', type=str, default='')
    parser.add_argument('--gt_mask_dir', type=str, default='')

    parser.add_argument('--name', type=str, default='pix2pix_dehaze41_1_1_public_default_unet_l1_100_BV_100')
    parser.add_argument('--test_name', type=str, default='test_latest_iter100')
    # parser.add_argument('--pretrain_model', type=str, default='seg_s2.pth')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--is_framework', action='store_true')
    parser.add_argument('--is_fake_B', action='store_true')
    parser.add_argument('--is_fake_TB', action='store_true')

    parser.add_argument('--save_segmentation', action='store_true')


    args = parser.parse_args()
    print(args)
    # TODO: 做一个TestDataset，把gt也load进来，直接评价，并且保存

    cudnn.benchmark = True
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    if args.is_framework:
        data_dir = '{}/{}/{}/images'.format(args.data_root, args.name, args.test_name)
        seg_output_dir = os.path.dirname(data_dir)
    else:
        data_dir = args.data_dir
        seg_output_dir = os.path.dirname(data_dir)

    if args.gt_dir != '':
        gt_dir_list = ['datasets/fiq_seg/target_gt_s2', 'datasets/fiq_seg/target_gt']
        # gt_dir_list = ['/images/liuhaofeng/Project/pixDA_GM/datasets/cataract_0830/target_seg_s2',
        #                '/images/liuhaofeng/Project/pixDA_GM/datasets/cataract_0830/target_seg_s3']
    else:
        gt_dir_list = ['', '']
    for seg_model, gt_dir in zip(['seg_s2.pth', 'seg_s3.pth'], gt_dir_list):

        inference_loader, model = eval_seg_init(seg_model, data_dir, args.mask_dir, gt_dir, args.gt_mask_dir,
                                                args.is_fake_B, args.is_fake_TB, device)
        if args.save_segmentation:
            seg_dir = os.path.join(seg_output_dir, seg_model.split('.')[0])
        else:
            seg_dir = ''
        print('saving the segmentation result in {}'.format(seg_dir))
        metrics = eval_seg_run(inference_loader, model, gt_dir, device, output_dir=seg_dir)
        if gt_dir != '':
            with open('model_eval/segmentation/result.csv', 'a', newline='') as f:
                f.write('{},{},'.format(args.data_dir, args.name))
                for k, v in metrics.items():
                    f.write('{:.3f},'.format(v))
                f.write('\n')
            print(metrics)
        model = None
        # torch.cuda.empty_cache()



