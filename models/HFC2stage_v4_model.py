# -*- coding: UTF-8 -*-
"""
@Function:
@File: hfc2stage.py
@Date: 2021/7/29 18:54 
@Author: Hever
"""
import torch
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    return (hfc + 1) * mask - 1


class HFC2StageV4Model(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired images.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test_total phase. You can use this flag to add training-specific or test_total-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use images buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_S1_L1H', type=float, default=100.0)
            parser.add_argument('--lambda_S2_L1', type=float, default=100.0)
            parser.add_argument('--lambda_S2_L1H', type=float, default=100.0)
            parser.add_argument('--sub_low_ratio', type=float, default=1.0, help='weight for L1L loss')
            parser.add_argument('--is_clamp', action='store_true')


        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_S1_L1H', 'G_S2_L1', 'G_S2_L1H']
        # specify the images you want to save/display. The training/test_total scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_SA_RBG', 'real_SA_gray', 'real_SAH', 'fake_S1_SRGB', 'fake_S1_Sgray',
        #                      'fake_S2_S_RGB', 'fake_S2_S_gray', 'real_SB', 'real_SBH',
        #                      'real_TA', 'real_TAH', 'fake_S1_T_RGB', 'fake_S1_T_gray', 'fake_S2_T_RGB', 'fake_S2_T_gray']
        self.visual_names = ['real_SA', 'real_SAG', 'fake_S1_SRGB', 'fake_S1_SG',
                             'fake_SB', 'fake_S2_SG',
                             'real_SB', 'real_SBG',
                             'real_TA',
                             'fake_TB', 'fake_S2_TG']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['GH', 'GL']

        if not self.isTrain:
            self.visual_names = ['real_TA', 'fake_S1_TH', 'fake_S2_T', 'fake_S2_TH']

        # define networks (both generator and discriminator)
        self.netGH = networks.define_G(4, 4, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netGL = networks.define_G(8, 4, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.hfc_filter = HFCFilter(21, 20, sub_low_ratio=opt.sub_low_ratio, sub_mask=True, is_clamp=opt.is_clamp).to(self.device)

        # if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        #     self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
        #                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_GH = torch.optim.Adam(self.netGH.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_GL = torch.optim.Adam(self.netGL.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_GH)
            self.optimizers.append(self.optimizer_GL)

    def set_input(self, input, isTrain=None):
        """
        处理输入
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
        self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
        self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
        self.real_SAG = input['SAG' if AtoB else 'SBG'].to(self.device)
        self.real_SBG = input['SBG' if AtoB else 'SAG'].to(self.device)
        self.real_TAG = input['TAG'].to(self.device)

        self.S_mask = input['S_mask'].to(self.device)
        self.T_mask = input['T_mask'].to(self.device)

        self.image_paths = input['TA_path']

        self.real_SA_input = torch.cat([self.real_SA, self.real_SAG], dim=1)
        self.real_SB_input = torch.cat([self.real_SB, self.real_SBG], dim=1)
        self.real_TA_input = torch.cat([self.real_TA, self.real_TAG], dim=1)
        self.real_SAH_input = hfc_mul_mask(self.hfc_filter, self.real_SA_input, self.S_mask)
        self.real_SBH_input = hfc_mul_mask(self.hfc_filter, self.real_SB_input, self.S_mask)
        self.real_TAH_input = hfc_mul_mask(self.hfc_filter, self.real_TA_input, self.T_mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""

        self.fake_S1_output = self.netGH(self.real_SAH_input)  # G(A)
        self.fake_S1_output = (self.fake_S1_output + 1) * self.S_mask - 1
        self.fake_S1_SRGB = self.fake_S1_output[:, :3]
        self.fake_S1_SG = self.fake_S1_output[:, 3:]


        # 反向传播不会对S1有影响
        self.fake_SB_output = self.netGL(torch.cat([self.real_SA_input, self.fake_S1_output.detach()], dim=1))  # G(A)
        self.fake_SBH_output = hfc_mul_mask(self.hfc_filter, self.fake_SB_output, self.S_mask)
        self.fake_SB = self.fake_SB_output[:, :3]
        self.fake_S2_SG =self.fake_SB_output[:, 3:]

        self.fake_T1_output = self.netGH(self.real_TAH_input)  # G(A)
        self.fake_T1_output = (self.fake_T1_output + 1) * self.T_mask - 1

        # 反向传播不会对S1有影响
        self.fake_TB_output = self.netGL(torch.cat([self.real_TA_input, self.fake_T1_output.detach()], dim=1))  # G(A)
        self.fake_TBH = hfc_mul_mask(self.hfc_filter, self.fake_TB_output, self.T_mask)
        self.fake_TB = self.fake_TB_output[:, :3]
        self.fake_S2_TG = self.fake_TB_output[:, 3:]

    def backward_GH(self):
        self.loss_G_S1_L1H = self.criterionL1(self.fake_S1_output, self.real_SBH_input) * self.opt.lambda_S1_L1H
        self.loss_G_S1_L1H.backward()

    def backward_GL(self):
        self.loss_G_S2_L1 = self.criterionL1(self.fake_SB_output, self.real_SB_input) * self.opt.lambda_S2_L1
        self.loss_G_S2_L1H = self.criterionL1(self.fake_SBH_output, self.real_SBH_input) * self.opt.lambda_S2_L1H
        self.loss_G_S2 = self.loss_G_S2_L1 + self.loss_G_S2_L1H

        self.loss_G_S2.backward()

    def optimize_parameters(self):
        self.set_requires_grad([self.netGL, self.netGH], True)  # D requires no gradients when optimizing G
        self.forward()                   # compute fake images: G(A)
        # # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights
        # # update G
        self.set_requires_grad(self.netGL, True)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netGH, False)  # D requires no gradients when optimizing G
        self.optimizer_GL.zero_grad()        # set G's gradients to zero
        self.backward_GL()                   # calculate graidents for G
        self.optimizer_GL.step()             # udpate G's weights


        self.set_requires_grad(self.netGL, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netGH, True)  # D requires no gradients when optimizing G
        self.optimizer_GH.zero_grad()  # set G's gradients to zero
        self.backward_GH()  # calculate graidents for G
        self.optimizer_GH.step()  # udpate G's weights
