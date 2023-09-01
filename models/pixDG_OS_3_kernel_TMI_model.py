# -*- coding: UTF-8 -*-
"""
@Function:从two-stage到one-stage，建立简化模型
@File: DG_one_model.py
@Date: 2021/9/14 20:45 
@Author: Hever
"""
# -*- coding: UTF-8 -*-
import torch
import itertools
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    # return hfc
    return (hfc + 1) * mask - 1
    # return images

def hfc_mul_mask_list(hfc_filter_list, image, mask):
    res_list = []
    for hfc_filter in hfc_filter_list:
        hfc = hfc_filter(image, mask)
        res = (hfc + 1) * mask - 1
        res_list.append(res)
    concat_res = torch.cat(res_list, dim=1)
    return concat_res

class PixDGOS3KernelTMIModel(BaseModel):
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
        parser.set_defaults(norm='instance', netG='unet_combine_2layer', dataset_mode='aligned', no_dropout=True,
                            output_nc=3)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0)
            parser.add_argument('--lambda_L1H', type=float, default=100.0)
            parser.add_argument('--lambda_L1_HFC', type=float, default=100.0)
            # parser.add_argument('--lambda_L1_idt', type=float, default=5.0)
            # parser.add_argument('--lambda_L1H_idt', type=float, default=5.0)
        parser.add_argument('--num_of_filter', type=int, default=3)
        parser.add_argument('--filters_width_list', nargs='+', type=int, default=[9, 19, 29])
        parser.add_argument('--nsig_list', nargs='+', type=float, default=[3.0, 5.0, 9.0])
        parser.add_argument('--sub_low_ratio', type=float, default=1.0, help='weight for L1L loss')
        # parser.add_argument('--is_clamp', action='store_true')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1H', 'G_L1', 'G_L1_HFC', 'G']

        self.visual_names_train = ['real_SA', 'real_SAH', 'fake_SBH', 'fake_SB', 'fake_SB_HFC',
                                   'real_SB', 'real_SBH', ]
        self.visual_names_test = ['real_TA', 'real_TAH', 'fake_TBH', 'fake_TB', 'fake_TB_HFC']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
            self.visual_names = self.visual_names_train
        else:
            self.model_names = ['G']
            self.visual_names = self.visual_names_test

        # define networks (both generator and discriminator)
        opt.input_nc = 3 * len(opt.filters_width_list)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.hfc_filter = HFCFilter(opt.filter_width, opt.nsig, sub_low_ratio=opt.sub_low_ratio, sub_mask=True, is_clamp=True).to(self.device)
        # 设置3个不同参数的高斯卷积
        self.hfc_filter_list = [
            HFCFilter(w, s, sub_low_ratio=opt.sub_low_ratio, sub_mask=True, is_clamp=True).to(self.device)
            for w, s in zip(opt.filters_width_list, opt.nsig_list)
        ]
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input, isTrain=None):
        """
        处理输入
        """
        AtoB = self.opt.direction == 'AtoB'
        if not self.isTrain or isTrain is not None:
            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.real_TAH = hfc_mul_mask_list(self.hfc_filter_list, self.real_TA, self.T_mask)
            # self.real_TAH = hfc_mul_mask_list(self.hfc_filter_list, self.real_TA, self.T_mask)
            self.image_paths = input['TA_path']
        else:
            self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
            self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
            self.S_mask = input['S_mask'].to(self.device)
            self.image_paths = input['SA_path']
            self.real_SAH = hfc_mul_mask_list(self.hfc_filter_list, self.real_SA, self.S_mask) # torch.Size([8, 9, 256, 256])
            self.real_SBH = hfc_mul_mask_list(self.hfc_filter_list, self.real_SB, self.S_mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        self.fake_SBH, self.fake_SB = self.netG(self.real_SAH)
        self.fake_SBH = (self.fake_SBH + 1) * self.S_mask - 1
        self.fake_SB = (self.fake_SB + 1) * self.S_mask - 1
        self.fake_SB_HFC = hfc_mul_mask_list(self.hfc_filter_list, self.fake_SB, self.S_mask)

    def compute_visuals(self):
        if self.isTrain:
            self.real_SAH = self.real_SAH[:, 6:, :, :]
            self.real_SBH = self.real_SBH[:, 6:, :, :]
            self.fake_SBH = self.fake_SBH[:, 6:, :, :]
            self.fake_SB_HFC = self.fake_SB_HFC[:, 6:, :, :]
        else:
            self.real_TAH = self.real_TAH[:, 6:, :, :]
            # self.real_TBH = self.real_TBH[:, 6:, :, :]
            self.fake_TBH = self.fake_TBH[:, 6:, :, :]
            self.fake_TB_HFC = self.fake_TB_HFC[:, 6:, :, :]

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            # 为了可视化，不会对网络进行操作
            self.fake_TBH, self.fake_TB = self.netG(self.real_TAH)
            self.fake_TBH = (self.fake_TBH + 1) * self.T_mask - 1
            self.fake_TB = (self.fake_TB + 1) * self.T_mask - 1
            self.fake_TB_HFC = hfc_mul_mask_list(self.hfc_filter_list, self.fake_TB, self.T_mask)
            # self.fake_TBH = self.hfc_filter(self.fake_TB_HFC, self.T_mask)

            self.compute_visuals()
            pass

    def train(self):
        """Make models eval mode during test_total time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        # pred_fake_SAB = self.netDPH(self.fake_S1_AB.detach())
        # fake_SAHSH = torch.cat((self.real_SAH, self.fake_SH), 1)
        # pred_fake = self.netDPH(fake_SAHSH)
        #
        # self.loss_G_DPH = self.criterionGAN(pred_fake, True) * self.opt.lambda_G_DPH
        # self.loss_G_S1_DPH = 0
        # LR
        self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1
        # LH
        self.loss_G_L1H = self.criterionL1(self.fake_SBH, self.real_SBH) * self.opt.lambda_L1H
        # ！！！Lcyc
        self.loss_G_L1_HFC = self.criterionL1(self.fake_SB_HFC, self.real_SBH) * self.opt.lambda_L1_HFC
        # self.loss_G_L1_HFC += self.criterionL1(self.fake_SB_HFC, self.fake_SBH.detach()) * self.opt.lambda_L1_HFC
        # self.loss_G_L1_HFC = self.criterionL1(self.fake_SB_HFC, self.fake_SBH.detach()) * self.opt.lambda_L1_HFC

        self.loss_G = self.loss_G_L1 + self.loss_G_L1H + self.loss_G_L1_HFC
        self.loss_G.backward()


    def optimize_parameters(self):
        # self.set_requires_grad([self.netG], True)  # D requires no gradients when optimizing G
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        # self.compute_visuals()

