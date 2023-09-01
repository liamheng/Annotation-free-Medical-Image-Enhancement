# -*- coding: UTF-8 -*-
"""

"""
import torch
import itertools
from .base_model import BaseModel
from . import networks
from util import util
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    # return hfc
    return (hfc + 1) * mask - 1
    # return images

# s
class CofeNetModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='cofenet')
        if is_train:
            parser.set_defaults(pool_size=0)
            parser.add_argument('--lambda_content', type=float, default=1)
            parser.add_argument('--lambda_seg', type=float, default=0.1)
            parser.add_argument('--lambda_art', type=float, default=0.1)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G_seg', 'G_art', 'G_content_half', 'G']

        self.visual_names_train = ['real_A', 'fake_B', 'real_B',  'real_B_seg', 'fake_B_seg', 'real_artifact', 'fake_artifact']
        # self.visual_names_test = ['real_A', 'real_AH', 'fake_BH', 'fake_B', 'fake_B_HFC']
        self.visual_names_test = ['real_A', 'fake_B']

        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
            self.visual_names = self.visual_names_train
        else:
            self.model_names = ['G']
            self.visual_names = self.visual_names_test

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # define loss functions
            self.criterionContent = torch.nn.L1Loss()
            self.criterionSegment = torch.nn.BCELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)






    def set_input(self, input, isTrain=None):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['SA' if AtoB else 'SB'].to(self.device)
        self.real_A_half = torch.nn.functional.interpolate(self.real_A, size=(128, 128), mode='bicubic')
        self.real_B = input['SB' if AtoB else 'SA'].to(self.device)
        self.real_B_half = torch.nn.functional.interpolate(self.real_B, size=(128, 128), mode='bicubic')
        self.S_mask = input['S_mask'].to(self.device)
        self.A_mask = self.S_mask
        self.B_mask = self.S_mask
        if isTrain:
            self.real_B_seg = input['B_seg'].to(self.device)
            self.real_B_seg_half = torch.nn.functional.interpolate(self.real_B_seg, size=(128, 128), mode='nearest')
            self.real_artifact = torch.zeros_like(self.S_mask)
            # self.real_artifact = self.S_mask
            self.real_artifact_half = torch.nn.functional.interpolate(self.real_artifact, size=(128, 128), mode='nearest')
        if not self.isTrain or isTrain is not None:
            self.isTrain = False
            self.real_TA = input['TA'].to(self.device)
            self.A_mask = self.S_mask
        self.image_paths = input['TA_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        self.fake_B, self.fake_B_half, self.fake_B_seg, self.fake_artifact = self.netG(self.real_A, self.real_A_half, self.real_A)
        self.fake_B = (self.fake_B + 1) * self.A_mask - 1

    def compute_visuals(self):
        self.fake_artifact = torch.nn.functional.interpolate(self.fake_artifact, size=(256, 256), mode='nearest')

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            self.fake_B, self.fake_B_half, self.fake_B_seg, self.fake_artifact = self.netG(self.real_A, self.real_A_half, self.real_A)
            self.fake_B = (self.fake_B + 1) * self.A_mask - 1
            self.fake_TB = self.fake_B
            # self.compute_visuals()


    def train(self):
        """Make models eval mode during test_total time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        # LR
        self.loss_G_L1 = self.criterionContent(self.fake_B, self.real_B) * self.opt.lambda_content
        self.loss_G_seg = self.criterionSegment(self.fake_B_seg, self.real_B_seg) * self.opt.lambda_seg
        self.loss_G_art = self.criterionContent(self.fake_artifact, self.real_artifact_half) * self.opt.lambda_art

        self.loss_G_content_half = self.criterionContent(self.fake_B_half, self.real_B_half) * self.opt.lambda_content
        # self.loss_G_art_content_half = self.criterionContent(self.fake_B_art_half,
        #                                            self.real_B_art_half) * self.opt.lambda_art


        self.loss_G = self.loss_G_L1 + self.loss_G_seg + self.loss_G_art + self.loss_G_content_half
        self.loss_G.backward()


    def optimize_parameters(self):
        # self.set_requires_grad([self.netG], True)  # D requires no gradients when optimizing G
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.compute_visuals()

