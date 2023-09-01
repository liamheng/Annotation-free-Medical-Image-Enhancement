# -*- coding: UTF-8 -*-
"""
@Function:对比pixDGV，该模型会添加了consistency的HFC loss

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


class pixDGVConsistencyModel(BaseModel):
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
            parser.add_argument('--lambda_L1H', type=float, default=100.0)
            parser.add_argument('--lambda_L1H_consistency', type=float, default=100.0)

            parser.add_argument('--lambda_L1', type=float, default=100.0)
            parser.add_argument('--lambda_G_DP', type=float, default=1, help='weight for DD')
            parser.add_argument('--lambda_G_DPH', type=float, default=1.0, help='weight for DD')

            parser.add_argument('--sub_low_ratio', type=float, default=1.0, help='weight for L1L loss')
            parser.add_argument('--is_clamp', action='store_true')
            parser.add_argument('--is_cos_sim', action='store_true')
            parser.add_argument('--lambda_cos_sim', type=float, default=20.0, help='weight for L1L loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['L1', 'L1H', 'L1H_consistency', 'G',
                           'G_DP', 'G_DPH', 'DP', 'DPH']
        if self.opt.is_cos_sim:
            self.loss_names += ['cos_sim']
        self.visual_names_train = ['real_SA', 'real_SAH', 'fake_SB', 'fake_SBH', 'real_SB', 'real_SBH']
        self.visual_names_test = ['real_TA', 'fake_TB', 'fake_TBH', 'real_TAH']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'DP', 'DPH']
            self.visual_names = self.visual_names_train
        else:
            self.model_names = ['G']
            self.visual_names = self.visual_names_test

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.hfc_filter = HFCFilter(21, 20, sub_low_ratio=opt.sub_low_ratio, sub_mask=True, is_clamp=opt.is_clamp).to(self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netDP = networks.define_D(opt.input_nc * 2, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDPH = networks.define_D(opt.input_nc * 2, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netDP.parameters(), self.netDPH.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, isTrain=None):
        """
        处理输入
        """
        AtoB = self.opt.direction == 'AtoB'
        if not self.isTrain or isTrain is not None:
            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.real_TAH = hfc_mul_mask(self.hfc_filter, self.real_TA, self.T_mask)
            self.image_paths = input['TA_path']
        else:
            self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
            self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
            self.S_mask = input['S_mask'].to(self.device)
            self.image_paths = input['SA_path']
            self.real_SAH = hfc_mul_mask(self.hfc_filter, self.real_SA, self.S_mask)
            self.real_SBH = hfc_mul_mask(self.hfc_filter, self.real_SB, self.S_mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        if self.opt.is_cos_sim:
            self.fake_SB, self.fake_SA_vector = self.netG(self.real_SAH)
        else:
            self.fake_SB = self.netG(self.real_SAH)

        self.fake_SB = (self.fake_SB + 1) * self.S_mask - 1
        self.fake_SBH = hfc_mul_mask(self.hfc_filter, self.fake_SB, self.S_mask)
        if self.opt.is_cos_sim:
            self.real_SB_vector = self.netG(self.real_SB, only_vector=True)
        # # S1是高频+高频
        # self.fake_SAB = torch.cat((self.real_SA, self.fake_SB), 1)
        # self.fake_SABH = torch.cat((self.real_SAH, self.fake_SBH), 1)

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            # 为了可视化，不会对网络进行操作
            if self.opt.is_cos_sim:
                self.fake_TB, self.hidden_vector = self.netG(self.real_TAH)  # G(A)
            else:
                self.fake_TB = self.netG(self.real_TAH)  # G(A)

            self.fake_TB = (self.fake_TB + 1) * self.T_mask - 1
            self.fake_TBH = hfc_mul_mask(self.hfc_filter, self.fake_TB, self.T_mask)
            self.compute_visuals()

    def train(self):
        """Make models eval mode during test_total time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_DP(self):
        fake_SASB = torch.cat((self.real_SA, self.fake_SB), 1)
        real_SASB = torch.cat((self.real_SA, self.real_SB), 1)

        pred_fake = self.netDP(fake_SASB.detach())
        self.loss_DP_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_G_DP

        pred_real = self.netDPH(real_SASB.detach())
        self.loss_DP_real = self.criterionGAN(pred_real, True) * self.opt.lambda_G_DP

        self.loss_DP = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_DP.backward()


    def backward_DPH(self):
        fake_SAHSBH = torch.cat((self.real_SAH, self.fake_SBH), 1)
        real_SAHSBH = torch.cat((self.real_SAH, self.real_SBH), 1)

        pred_fake = self.netDPH(fake_SAHSBH.detach())
        self.loss_DPH_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_G_DPH

        pred_real = self.netDPH(real_SAHSBH.detach())
        self.loss_DPH_real = self.criterionGAN(pred_real, True) * self.opt.lambda_G_DPH

        self.loss_DPH = (self.loss_DPH_fake + self.loss_DPH_real) * 0.5
        self.loss_DPH.backward()

    def backward_G(self):
        # pred_fake_SAB = self.netDPH(self.fake_S1_AB.detach())
        fake_SASB = torch.cat((self.real_SA, self.fake_SB), 1)
        pred_fake = self.netDP(fake_SASB)

        fake_SAHSBH = torch.cat((self.real_SAH, self.fake_SBH), 1)
        pred_fakeH = self.netDPH(fake_SAHSBH)

        self.loss_G_DP = self.criterionGAN(pred_fake, True) * self.opt.lambda_G_DP
        self.loss_G_DPH = self.criterionGAN(pred_fakeH, True) * self.opt.lambda_G_DPH

        # self.loss_G_S1_DPH = 0
        self.loss_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1
        self.loss_L1H = self.criterionL1(self.fake_SBH, self.real_SBH) * self.opt.lambda_L1H
        self.loss_L1H_consistency = self.criterionL1(self.fake_SBH, self.real_SAH) * self.opt.lambda_L1H_consistency

        self.loss_G = self.loss_G_DPH + self.loss_G_DP + self.loss_L1H + self.loss_L1 + self.loss_L1H_consistency
        if self.opt.is_cos_sim:
            self.fake_SA_vector = self.fake_SA_vector.reshape([self.fake_SA_vector.shape[0], -1])
            self.real_SB_vector = self.real_SB_vector.reshape([self.fake_SA_vector.shape[0], -1])

            self.loss_cos_sim = (1 - torch.cosine_similarity(x1=self.fake_SA_vector, x2=self.real_SB_vector, dim=1))
            self.loss_cos_sim = torch.sum(self.loss_cos_sim) / self.fake_SA_vector.shape[0] * self.opt.lambda_cos_sim
            # self.loss_cos_sim = self.criterionL1(self.fake_SA_vector, self.real_SB_vector) * self.opt.lambda_cos_sim
            self.loss_G += self.loss_cos_sim
        self.loss_G.backward()


    def optimize_parameters(self):
        self.set_requires_grad([self.netG], True)  # D requires no gradients when optimizing G
        self.forward()                   # compute fake images: G(A)

        # # update D
        self.set_requires_grad([self.netDP, self.netDPH], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_DP()  # calculate gradients for D_A
        self.backward_DPH()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # update G
        self.set_requires_grad([self.netDP, self.netDPH], False)

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

