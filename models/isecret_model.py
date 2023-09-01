# -*- coding: UTF-8 -*-
"""
@Function:
@File: isecret_model.py
@Date: 2022/7/11 16:20 
@Author: Hever
"""
import torch
from .base_model import BaseModel
from . import networks
from models.backbone.isecret.backbone import PatchNCELoss, ISLoss, LSGANLoss

def mul_mask(image, mask):
    return (image + 1) * mask - 1

class ISecretModel(BaseModel):
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
        parser.set_defaults(norm='instance', netG='isecret_backbone', dataset_mode='aligned', pool_size=0, netD='basic')

        if is_train:
        #     parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--lambda_icc', type=float, default=1.0, help='weight for G loss')
            parser.add_argument('--lambda_is', type=float, default=1.0, help='weight for G loss')
            parser.add_argument('--lambda_idt', type=float, default=1.0, help='the weight of the idt-loss')

        # parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        # parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        # parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        # parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
        #                     help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        # parser.add_argument('--nce_layers', type=str, default='1,5,9,11,15,18,20', help='compute NCE loss on which layers')
        parser.add_argument('--nce_layers', nargs='+', type=int, default=[1,5,9,11,15,18,20], help='compute NCE loss on which layers')

        # parser.add_argument('--nce_includes_all_negatives_from_minibatch',
        #                     type=util.str2bool, nargs='?', const=True, default=False,
        #                     help='(used for single images translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch', action='store_true')
        # parser.add_argument('--lambda_gan', type=float, default=1.0, help='the weight of the gan-loss')
        # parser.add_argument('--lambda_icc', type=float, default=1.0, help='the weight of the icc-loss')


        # parser.add_argument('--flip_equivariance',
        #                     type=util.str2bool, nargs='?', const=True, default=False,
        #                     help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no images pooling
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names_train = ['real_SA', 'fake_SB', 'real_SAU',
                                   'fake_SBU',
                                   # 'fake_SB_importance_rec_vis',
                                   # 'fake_SBU_importance_vis', 'fake_SB_importance_vis'
                                   ]
        self.visual_names_test = ['real_TA', 'fake_TB',
                                  # 'fake_TB_importance_vis'
                                  ]
        self.loss_names = ['supervised', 'unsupervised', 'D']

        self.nce_layers = self.opt.nce_layers
        # self.n
        if self.isTrain:
            self.model_names = ['G', 'D', 'F']
            self.visual_names = self.visual_names_train
        else:  # during test_total time, only load G
            self.model_names = ['G']
            self.visual_names = self.visual_names_test
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.norm, not opt.no_dropout, opt.init_type,
                                          opt.init_gain,
                                          False, self.gpu_ids, opt)
            self._nce_losses = []
            for nce_layer in self.nce_layers:
                self._nce_losses.append(PatchNCELoss(opt).to(self.device))
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.gan_loss = LSGANLoss()
            # self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.optimizers.append(self.optimizer_F)

            self.rec_loss = ISLoss()


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_SA.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_SA = self.real_SA[:bs_per_gpu]
        self.real_SB = self.real_SB[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            feat_k, _ = self.netG(self.real_SA, layers=self.opt.nce_layers)
            self.netF(feat_k, self.opt.netF_nc, None)  # Initialize
            # self.set_requires_grad([self.netG], False)
            # self.loss_D = self.gan_loss.update_d(self.netD, self.real_SB, self.fake_SB)
            # self.loss_D.backward()
            # self.compute_D_loss().backward()  # calculate gradients for D
            # self.compute_G_loss().backward()  # calculate graidents for G
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_F)
            self.icc_loss = self.compute_nce_loss
            self.idt_loss = self.compute_nce_loss

    def set_input(self, input, isTrain=None):
        AtoB = self.opt.direction == 'AtoB'
        if not self.isTrain or isTrain == False:
            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.image_paths = input['TA_path']
        else:
            self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
            self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
            self.real_SAU = input['SAU'].to(self.device)

            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.S_mask = input['S_mask'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.image_paths = input['SA_path' if AtoB else 'SB_path']


    def _vis_importance(self, importance):
        importance = torch.exp(importance)
        for idx in range(importance.shape[0]):
            importance[idx, ...] = (importance[idx, ...] - torch.min(importance[idx, ...])) \
                                   / (torch.max(importance[idx, ...]) - torch.min(importance[idx, ...]))
        return importance

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        pass

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            self.fake_TB, importance_fake_TB = self.netG(self.real_TA, need_importance=True)  # G(A)
            self.fake_TB = mul_mask(self.fake_TB, self.T_mask)
            # self.fake_TB_importance_vis = self._vis_importance(importance_fake_TB)
            # self.fake_TB = self.fake_B

    def train(self):
        """Make models eval mode during test_total time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def compute_nce_loss(self, source, target, weight_map=None):
        feat_q, _ = self.netG(target, layers=self.opt.nce_layers)
        feat_k, _  = self.netG(source, layers=self.opt.nce_layers)
        if weight_map is None:
            feat_k_pool, sample_ids = self.netF(feat_k, self.opt.netF_nc)
            feat_q_pool, _ = self.netF(feat_q, self.opt.netF_nc, sample_ids)
            nce_loss = 0.
            for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self._nce_losses, self.opt.nce_layers):
                nce_loss += crit(f_q, f_k).mean()
        else:
            feat_k_pool, sample_ids, weight_sample = self.netF(feat_k, self.opt.netF_nc, None, weight_map=weight_map)
            feat_q_pool, _ = self.netF(feat_q, self.opt.netF_nc, sample_ids)
            nce_loss = 0.
            for f_q, f_k, crit, weight, nce_layer in zip(feat_q_pool, feat_k_pool, self._nce_losses, weight_sample, self.opt.nce_layers):
                nce_loss += crit(f_q, f_k, weight).mean()
        nce_loss /= len(self.opt.nce_layers)
        return nce_loss

    def _train_supervised(self):
        self.fake_SB, importance_rec = self.netG(self.real_SA, need_importance=True)
        self.loss_is = self.rec_loss(self.fake_SB, self.real_SB,
                                          importance_rec) * self.opt.lambda_is
        # self.fake_SB_importance_rec_vis = self._vis_importance(importance_rec)
        self.loss_supervised = self.loss_is
        self.loss_supervised.backward()
        pass

    def _train_unsupervised(self):
        # Forward network
        real = torch.cat((self.real_SA, self.real_SAU), dim=0)
        fake, importance = self.netG(real, need_importance=True)
        importance_fake_SB, importance_fake_SBU = importance.chunk(2, dim=0)
        importance_fake_SBU = importance_fake_SBU.detach()

        # Visualize importance map
        # self.fake_SB_importance_vis = self._vis_importance(importance_fake_SB)
        # self.fake_SBU_importance_vis = self._vis_importance(importance_fake_SBU)
        self.fake_SB, self.fake_SBU = fake.chunk(2, dim=0)

        self.loss_icc = self.icc_loss(self.real_SAU, self.fake_SBU, importance_fake_SBU) * self.opt.lambda_icc
        self.loss_idt = self.idt_loss(self.real_SA, self.fake_SB) * self.opt.lambda_idt
        self.loss_gan = self.gan_loss.update_g(self.netD, self.fake_SB) * self.opt.lambda_gan
        # self.loss_icc = 0
        # self.loss_idt = 0
        # self.loss_gan = 0

        self.loss_unsupervised = self.loss_icc + self.loss_idt + self.loss_gan
        self.loss_unsupervised.backward()
        pass

        # return losses, meta

    def optimize_D(self):
        self.set_requires_grad([self.netG], False)
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.netD.zero_grad()
        self.loss_D = self.gan_loss.update_d(self.netD, self.real_SB, self.fake_SB)

        self.loss_D.backward()
        self.optimizer_D.step()
        self.set_requires_grad([self.netG], True)
        pass
        # return losses

    def optimize_parameters(self):
        # train gen
        # forward supervised
        # self.forward()  # supervised, unspervised

        self.set_requires_grad(self.netD, False)
        self.netG.zero_grad()
        self.netF.zero_grad()
        self._train_supervised()
        self._train_unsupervised()
        self.optimizer_G.step()
        self.optimizer_F.step()

        self.optimize_D()  # calculate gradients for D

        pass