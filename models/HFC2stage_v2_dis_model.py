# -*- coding: UTF-8 -*-
"""
@Function:
@File: hfc2stage.py
@Date: 2021/7/29 18:54 
@Author: Hever
"""
import torch
import itertools
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    return (hfc + 1) * mask - 1


class HFC2StageV2DisModel(BaseModel):
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
            parser.add_argument('--lambda_G_S1_DPH', type=float, default=0.1, help='weight for DD')
            parser.add_argument('--lambda_G_S2_DPL', type=float, default=1.0, help='weight for DD')

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
        self.loss_names = ['G_S1_L1H', 'G_S2_L1', 'G_S2_L1H', 'DH', 'DL', 'G_S1_DPH', 'G_S2_DPL']
        # specify the images you want to save/display. The training/test_total scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_SA_RBG', 'real_SA_gray', 'real_SAH', 'fake_S1_SRGB', 'fake_S1_Sgray',
        #                      'fake_S2_S_RGB', 'fake_S2_S_gray', 'real_SB', 'real_SBH',
        #                      'real_TA', 'real_TAH', 'fake_S1_T_RGB', 'fake_S1_T_gray', 'fake_S2_T_RGB', 'fake_S2_T_gray']
        self.visual_names = ['real_SA', 'real_SAH', 'fake_S1_SH',
                             'fake_SB', 'fake_SBH', 'real_SB', 'real_SBH',
                             'real_TA', 'real_TAH', 'fake_S1_TH',
                             'fake_TB', 'fake_TBH']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['GH', 'GL', 'DPH', 'DPL']
        else:
            self.model_names = ['GH', 'GL']


        if not self.isTrain:
            self.visual_names = ['real_TA', 'fake_S1_TH', 'fake_S2_T', 'fake_S2_TH']

        # define networks (both generator and discriminator)
        self.netGH = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netGL = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.hfc_filter = HFCFilter(21, 20, sub_low_ratio=opt.sub_low_ratio, sub_mask=True, is_clamp=opt.is_clamp).to(self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netDPH = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDPL = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_GH = torch.optim.Adam(self.netGH.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_GL = torch.optim.Adam(self.netGL.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netDPH.parameters(), self.netDPL.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_GH)
            self.optimizers.append(self.optimizer_GL)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, isTrain=None):
        """
        处理输入
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
        self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
        self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)

        self.S_mask = input['S_mask'].to(self.device)
        self.T_mask = input['T_mask'].to(self.device)

        self.image_paths = input['TA_path']

        self.real_SAH = hfc_mul_mask(self.hfc_filter, self.real_SA, self.S_mask)
        self.real_SBH = hfc_mul_mask(self.hfc_filter, self.real_SB, self.S_mask)
        self.real_TAH = hfc_mul_mask(self.hfc_filter, self.real_TA, self.T_mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""

        self.fake_S1_SH = self.netGH(self.real_SAH)  # G(A)
        self.fake_S1_SH = (self.fake_S1_SH + 1) * self.S_mask - 1
        # S1是高频+高频
        # self.fake_S1_AB = torch.cat((self.real_SAH, self.fake_S1_SH), 1)
        # self.real_S1_AB = torch.cat((self.real_SAH, self.real_SBH), 1)

        # 反向传播不会对S1有影响
        self.fake_SB = self.netGL(self.fake_S1_SH.detach())  # G(A)
        self.fake_SBH = hfc_mul_mask(self.hfc_filter, self.fake_SB, self.S_mask)
        # S2是高频+RGB
        # self.fake_S2_AB = torch.cat((self.fake_S1_SH, self.fake_SB), 1)
        # self.real_S2_AB = torch.cat((self.fake_S1_SH, self.real_SB), 1)

        # 为了可视化，不会对网络进行操作
        self.fake_S1_TH = self.netGH(self.real_TAH)  # G(A)
        self.fake_S1_TH = (self.fake_S1_TH + 1) * self.T_mask - 1
        # 反向传播不会对S1有影响
        self.fake_TB = self.netGL(self.fake_S1_TH.detach())  # G(A)
        self.fake_TBH = hfc_mul_mask(self.hfc_filter, self.fake_TB, self.T_mask)

    def backward_DPH(self):
        fake_S1_AB = torch.cat((self.real_SAH, self.fake_S1_SH), 1)
        real_S1_AB = torch.cat((self.real_SAH, self.real_SBH), 1)
        pred_fake = self.netDPH(fake_S1_AB.detach())
        self.loss_DH_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netDPH(real_S1_AB.detach())
        self.loss_DH_real = self.criterionGAN(pred_real, True)

        self.loss_DH = (self.loss_DH_fake + self.loss_DH_real) * 0.5
        self.loss_DH.backward()

    def backward_DPL(self):
        fake_S2_AB = torch.cat((self.fake_S1_SH, self.fake_SB), 1)
        real_S2_AB = torch.cat((self.fake_S1_SH, self.real_SB), 1)
        pred_fake = self.netDPL(fake_S2_AB.detach())
        self.loss_DL_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netDPL(real_S2_AB.detach())
        self.loss_DL_real = self.criterionGAN(pred_real, True)

        self.loss_DL = (self.loss_DL_fake + self.loss_DL_real) * 0.5
        self.loss_DL.backward()

    def backward_GH(self):
        # pred_fake_SAB = self.netDPH(self.fake_S1_AB.detach())
        fake_S1_AB = torch.cat((self.real_SAH, self.fake_S1_SH), 1)
        pred_fake_SAB = self.netDPH(fake_S1_AB)

        self.loss_G_S1_DPH = self.criterionGAN(pred_fake_SAB, True) * self.opt.lambda_G_S1_DPH
        # self.loss_G_S1_DPH = 0

        self.loss_G_S1_L1H = self.criterionL1(self.fake_S1_SH, self.real_SBH) * self.opt.lambda_S1_L1H
        self.loss_G_S1 = self.loss_G_S1_DPH + self.loss_G_S1_L1H
        self.loss_G_S1.backward()

    def backward_GL(self):
        # pred_fake_SAB = self.netDPH(self.fake_S2_AB.detach())
        # !!!!!!!!!!!!!!!!!!!阻止S1_SH反向传播
        fake_S2_AB = torch.cat((self.fake_S1_SH.detach(), self.fake_SB), 1)
        pred_fake_SAB = self.netDPL(fake_S2_AB)

        self.loss_G_S2_DPL = self.criterionGAN(pred_fake_SAB, True) * self.opt.lambda_G_S2_DPL
        # self.loss_G_S2_DPL = 0

        self.loss_G_S2_L1 = self.criterionL1(self.fake_SB, self.real_SB.detach()) * self.opt.lambda_S2_L1
        self.loss_G_S2_L1H = self.criterionL1(self.fake_SBH, self.real_SBH.detach()) * self.opt.lambda_S2_L1H
        self.loss_G_S2 = self.loss_G_S2_DPL + self.loss_G_S2_L1 + self.loss_G_S2_L1H
        self.loss_G_S2.backward()

    def optimize_parameters(self):
        self.set_requires_grad([self.netGL, self.netGH], True)  # D requires no gradients when optimizing G
        self.forward()                   # compute fake images: G(A)

        # # update D
        self.set_requires_grad([self.netDPH, self.netDPL], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_DPH()  # calculate gradients for D_A
        self.backward_DPL()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # update G
        self.set_requires_grad([self.netDPH, self.netDPL], False)
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
