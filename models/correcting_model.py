# -*- coding: UTF-8 -*-
"""
@Function:
@File: correcting_model.py
@Date: 2021/7/15 10:53 
@Author: Hever
"""
import torch
import torch.nn.functional as F
from models.backbone.Fu_model import Fu_model
from .base_model import BaseModel


def define_fu_model(input_nc, output_nc, gpu_ids=[]):
    net = Fu_model(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net

def mul_mask(image, mask):
    return (image + 1) * mask - 1

class CorrectingModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', dataset_mode='aligned', dataroot='./dataset/cataract_fu')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_Lm', type=float, default=1, help='weight for L1 loss')
            parser.add_argument('--lambda_Lp', type=float, default=10, help='weight for G loss')
            parser.add_argument('--lambda_Lc', type=float, default=1, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['Lm1', 'Lm1', 'Lp2', 'Lp2', 'Lc2', 'Lc2']
        # specify the images you want to save/display. The training/test_total scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['input1', 'input2', 'input3', 'input4', 'gt_output_image1', 'gt_output_image2',
        #                      'gt_artifact_mask1', 'gt_artifact_mask2', 'pred_artifact_mask1',
        #                      'seg_mask1', 'final_output1', 'pred_artifact_mask2', 'seg_mask2', 'final_output2']
        self.visual_names = ['input1', 'input3', 'gt_output_image1',
                             'gt_artifact_mask1', 'pred_artifact_mask1',
                             'seg_mask1', 'fake_TB']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['Fu']
        # define networks (both generator and discriminator)
        self.netFu = define_fu_model(opt.input_nc, opt.output_nc, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionMSE = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_Fu = torch.optim.Adam(self.netFu.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Fu)
        else:
            self.visual_names = ['input1', 'input3', 'fake_TB']

    def set_input(self, input, isTrain=None):
        self.input1 = input['SA'].to(self.device)  # 256*256
        self.input2 = F.interpolate(self.input1, scale_factor=1 / 2, mode='bilinear', align_corners=True)  # 128*128
        self.input3 = F.interpolate(self.input2, scale_factor=2, mode='bilinear', align_corners=True)  # 256*256
        self.input4 = F.interpolate(self.input3, scale_factor=1 / 2, mode='bilinear', align_corners=True)  # 256*256
        self.gt_output_image1 = input['SB'].to(self.device)  # 256*256
        self.gt_output_image2 = F.interpolate(self.gt_output_image1, scale_factor=1 / 2, mode='bilinear', align_corners=True)  # 128*128
        # self.gt_artifact_mask1 = input['gt_artifact_mask'].to(self.device)  # 256*256
        # self.gt_artifact_mask2 = F.interpolate(self.gt_artifact_mask1, scale_factor=1 / 2, mode='bilinear', align_corners=True)  # 128*128
        self.gt_artifact_mask1 = -1 * torch.zeros_like(self.gt_output_image1)
        # self.gt_artifact_mask1 =
        self.gt_artifact_mask2 = F.interpolate(self.gt_artifact_mask1, scale_factor=1 / 2, mode='bilinear',
                                               align_corners=True)  # 128*128
        # self.gt_artifact_mask2 = F.interpolate(self.gt_artifact_mask1, scale_factor=1 / 2, mode='bilinear', align_corners=True)  # 64*64
        self.image_paths = input['SA_path']
        if not self.isTrain or isTrain is not None:
            self.input1 = input['TA'].to(self.device)
            self.input2 = F.interpolate(self.input1, scale_factor=1 / 2, mode='bilinear', align_corners=True)  # 128*128
            self.input3 = F.interpolate(self.input2, scale_factor=2, mode='bilinear', align_corners=True)  # 256*256
            self.image_paths = input['TA_path']
            self.T_mask = input['T_mask'].to(self.device)

    def forward(self):
        if self.isTrain:
            # 注意：为了测试方便，将final_output1改名为fake_TB
            self.pred_artifact_mask1, self.seg_mask1, self.fake_TB = self.netFu(self.input1, self.input3)
            self.pred_artifact_mask2, self.seg_mask2, self.final_output2 = self.netFu(self.input2, self.input4)
        else:
            _, _, self.fake_TB = self.netFu(self.input1, self.input3)
            self.fake_TB = mul_mask(self.fake_TB, self.T_mask)


    def backward_Fu(self):
        self.loss_Lm1 = self.criterionMSE(self.pred_artifact_mask1, self.gt_artifact_mask1) * self.opt.lambda_Lm
        self.loss_Lm2 = self.criterionMSE(self.pred_artifact_mask2, self.gt_artifact_mask2) * self.opt.lambda_Lm

        self.loss_Lc1 = self.criterionMSE(self.fake_TB, self.gt_output_image1) * self.opt.lambda_Lc
        self.loss_Lc2 = self.criterionMSE(self.final_output2, self.gt_output_image2) * self.opt.lambda_Lc

        self.loss_Lp1 = self.criterionMSE(self.pred_artifact_mask1 * self.fake_TB,
                                          self.pred_artifact_mask1 * self.gt_output_image1) * self.opt.lambda_Lp
        self.loss_Lp2 = self.criterionMSE(self.pred_artifact_mask2 * self.final_output2,
                                          self.pred_artifact_mask2 * self.gt_output_image2) * self.opt.lambda_Lp

        self.loss_G = (self.loss_Lm1 + self.loss_Lm1) \
                      + (self.loss_Lc1 + self.loss_Lc2) \
                      + (self.loss_Lp1 + self.loss_Lp2)


        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)

        self.optimizer_Fu.zero_grad()  # set D's gradients to zero
        self.backward_Fu()  # calculate gradients for D
        self.optimizer_Fu.step()  # update D's weights
