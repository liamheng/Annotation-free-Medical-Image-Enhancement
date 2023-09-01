# -*- coding: utf-8 -*-

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.backbone.still_gan_backbone import LuminanceLoss, StructureLoss
from .guided_filter_pytorch.sobel_filter import FourSobelFilter


def mul_mask(image, mask):
    return (image + 1) * mask - 1

class StillGANSingleSCRModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning images-to-images translation without paired images.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test_total phase. You can use this flag to add training-specific or test_total-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True, netG='resunet', pool_size=50, norm='instance')  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--Luminance_size1', type=int, default=48,
                                help='first window size for Luminance Loss')
            parser.add_argument('--Luminance_size2', type=int, default=96,
                                help='second window size for Luminance Loss')
            parser.add_argument('--Structure_size', type=int, default=11,
                                help='first window size for Structure Loss')
            parser.add_argument('--roi_size', type=int, default=192,
                                help='size of ROI for Luminance Loss and Structure Loss')
            parser.add_argument('--lambda_Luminance_B', type=float, default=0.1,
                                help='weight for Luminance Loss')
            parser.add_argument('--lambda_Structure', type=float, default=0.5,
                                help='weight for Structure Loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','cycle_sobel_A','cycle_sobel_B']
        # specify the images you want to save/display. The training/test_total scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A','real_A_sobel','fake_B_sobel']
        visual_names_B = ['real_B', 'fake_A', 'rec_B','real_B_sobel','fake_A_sobel']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        if self.isTrain and self.opt.lambda_Luminance_B >= 0.0:  # Illumination regularization is used
            self.loss_names.append('Luminance_B')
        if self.isTrain and self.opt.lambda_Structure >= 0.0:  # Structure loss is used
            self.loss_names.append('Structure_A')
            self.loss_names.append('Structure_B')

        self.visual_names = visual_names_A + visual_names_B + ['fake_TB']  # combine visualizations for A and B
        self.visual_names_test = ['real_A', 'fake_TB']
        self.visual_names_train = self.visual_names
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test_total time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create images buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create images buffer to store previously generated images
            self.fake_A_sobel_pool = ImagePool(opt.pool_size)  # create images buffer to store previously generated images
            self.fake_B_sobel_pool = ImagePool(opt.pool_size)  # create images buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionLuminance1 = LuminanceLoss(self.opt.Luminance_size1, self.opt.Luminance_size1, self.opt.roi_size).to(self.device)
            self.criterionLuminance2 = LuminanceLoss(self.opt.Luminance_size2, self.opt.Luminance_size2, self.opt.roi_size).to(self.device)
            self.criterionStructure = StructureLoss(channel=opt.output_nc, window_size=self.opt.Structure_size, crop_size=self.opt.roi_size).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.edge_filter_Four = FourSobelFilter(self.device)

    def set_input(self, input, isTrain=True):
        """Unpack input images from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the images itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'

        if isTrain:
            # if self.isTrain:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['SA' if AtoB else 'SB'].to(self.device)
            self.real_B = input['SB' if AtoB else 'SA'].to(self.device)
            self.image_paths = input['SA_path']
        else:
            self.real_A = input['TA'].to(self.device)
            self.real_B = input['TA'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.image_paths = input['TA_path']

        self.real_A_sobel = self.edge_filter_Four(self.real_A)
        self.real_B_sobel = self.edge_filter_Four(self.real_B)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        self.fake_B,self.fake_B_sobel = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A,self.rec_A_sobel = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A,self.fake_A_sobel = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B,self.rec_B_sobel = self.netG_A(self.fake_A)   # G_A(G_B(B))
        self.fake_TB = self.fake_B

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            # 为了可视化，不会对网络进行操作
            self.fake_B,_ = self.netG_A(self.real_A)  # G_A(A)
            # self.fake_B, _ = self.netG_B(self.real_A)  # G_A(A)
            # self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
            # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            # self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
            # self.fake_B = mul_mask(self.fake_B, self.T_mask)
            self.fake_TB = self.fake_B
            # self.fake_TBH = self.hfc_filter(self.fake_TB_HFC, self.T_mask)
            # self.compute_visuals()

    def train(self):
        """Make models eval mode during test_total time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # def backward_D_A_sobel(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     fake_B_sobel = self.fake_B_sobel_pool.query(self.fake_B_sobel)
    #     self.loss_D_A_sobel = self.backward_D_basic(self.netD_A_sobel, self.real_B_sobel, fake_B_sobel)
    #
    # def backward_D_B_sobel(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     fake_A_sobel = self.fake_A_sobel_pool.query(self.fake_A_sobel)
    #     self.loss_D_B_sobel = self.backward_D_basic(self.netD_B_sobel, self.real_A_sobel, fake_A_sobel)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_Luminance_B = self.opt.lambda_Luminance_B
        lambda_Structure = self.opt.lambda_Structure
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A,self.idt_A_sobel = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B,self.idt_B_sobel = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt


        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            self.loss_idt_A_sobel = 0
            self.loss_idt_B_sobel = 0
        if lambda_Luminance_B > 0:
            # Illumination Regularization
            self.loss_Luminance_B = (self.criterionLuminance1(self.fake_B) + self.criterionLuminance2(self.fake_B)) * lambda_B * lambda_Luminance_B / 2.0
        else:
            self.loss_Luminance_B = 0
        if lambda_Structure > 0:
            # Structure Loss
            self.loss_Structure_A = self.criterionStructure(self.real_B, self.fake_A) * lambda_A * lambda_Structure
            self.loss_Structure_B = self.criterionStructure(self.real_A, self.fake_B) * lambda_B * lambda_Structure
            # TODO
            # Structure Loss 修改了原版的
            # self.loss_Structure_A = self.criterionStructure(self.real_B, self.fake_B) * lambda_A * lambda_Structure
            # self.loss_Structure_B = self.criterionStructure(self.real_A, self.fake_A) * lambda_B * lambda_Structure
        else:
            self.loss_Structure_A = 0

            self.loss_Structure_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_cycle_sobel_A = self.criterionCycle(self.fake_A_sobel, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_sobel_B = self.criterionCycle(self.fake_B_sobel, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A \
                      + self.loss_idt_B + self.loss_Luminance_B + self.loss_Structure_A + self.loss_Structure_B \
                    + self.loss_cycle_sobel_A + self.loss_cycle_sobel_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
