import torch
from .base_model import BaseModel
from . import networks
from models.backbone.RFormer.RFormer import *
from models.networks import init_weights
import models.BSRI.losses as losses
import numpy as np
from models.backbone.RFormer.loss import CharbonnierLoss, PerceptualLoss, EdgeLoss
from models.backbone.RFormer.utils import MixUp_AUG
from torch.autograd import Variable


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def mul_mask(image, mask):
    return (image + 1) * mask - 1

class RFormerFixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for G loss')
            parser.add_argument('--lambda_char', type=float, default=1)
            parser.add_argument('--lambda_per', type=float, default=0.06)
            parser.add_argument('--lambda_edge', type=float, default=0.05)
            parser.add_argument('--lambda_patch', type=float, default=0.2)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_D = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_names = ['G', 'char', 'per', 'edge', 'G_patch', 'D_fake', 'D_real', 'D']

        self.visual_names_train = ['real_SA', 'real_SB',
                             'fake_SB']
        self.visual_names_test = ['real_TA', 'fake_TB']

        if self.isTrain:
            self.model_names = ['G', 'D']
            self.visual_names = self.visual_names_train

        else:  # during test time, only load G
            self.model_names = ['G']
            self.visual_names = self.visual_names_test

        self.netG = RFormer_G()
        self.netG = init_net(self.netG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = RFormer_D()
            self.netD = init_net(self.netD, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.lossc = CharbonnierLoss().to(self.device)
            self.lossp = PerceptualLoss().to(self.device)
            self.lossm = nn.MSELoss().to(self.device)
            self.losse = EdgeLoss().to(self.device)
            # self.real_labels_patch = Variable(torch.ones(opt.batch_size, 169) - 0.05).to(self.device)
            # self.fake_labels_patch = Variable(torch.zeros(opt.batch_size, 169)).to(self.device)
            self.real_labels_patch = Variable(torch.ones(opt.batch_size, 196) - 0.05).to(self.device)
            self.fake_labels_patch = Variable(torch.zeros(opt.batch_size, 196)).to(self.device)
            self.D_loss_sum_patch = 0
            self.mixup = MixUp_AUG()

    def set_input(self, input, isTrain=None):
        AtoB = self.opt.direction == 'AtoB'
        if not self.isTrain or isTrain == False:
            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.image_paths = input['TA_path']
        else:
            self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
            self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
            self.S_mask = input['S_mask'].to(self.device)
            self.image_paths = input['SA_path']
            if self.current_epoch > 5:
                self.real_SB, self.real_SA = self.mixup.aug(self.real_SB, self.real_SA)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_SB = self.netG(self.real_SA)  # G(SA)
        self.fake_SB = self.fake_SB.clamp(-1, 1)
        pass



    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            self.fake_TB = self.netG(self.real_TA)
        self.fake_TB = self.fake_TB.clamp(-1, 1)
        self.fake_TB = mul_mask(self.fake_TB, self.T_mask)

    def train(self):
        """Make models eval mode during test time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        # self.fake_SB = self.netG(self.real_SA)
        # self.fake_SB = self.fake_SB.clamp(-1, 1)

        fake_output = self.netD(self.fake_SB)
        real_output = self.netD(self.real_SB)
        if len(self.fake_labels_patch) > len(fake_output):
            self.loss_G_patch = self.lossm(fake_output, self.real_labels_patch[:len(real_output)])
        else:
            self.loss_G_patch = self.lossm(fake_output, self.real_labels_patch)
        self.loss_G_patch = self.loss_G_patch * self.opt.lambda_patch
        self.loss_char = self.lossc(self.fake_SB, self.real_SB) * self.opt.lambda_char
        self.loss_per = self.lossp(self.fake_SB, self.real_SB) * self.opt.lambda_per
        self.loss_edge = self.losse(self.fake_SB, self.real_SB) * self.opt.lambda_edge
        self.loss_G = self.loss_char + self.loss_per + self.loss_edge + self.loss_G_patch
        # self.loss_G = (self.loss_supervised + self.loss_consistency) * self.opt.lambda_G
        self.loss_G.backward()

    def backward_D(self):
        self.real_output = self.netD(self.real_SB)
        if len(self.real_labels_patch) > len(self.real_output):
            self.loss_D_real = self.lossm(self.real_output, self.real_labels_patch[:len(self.real_output)])
        else:
            self.loss_D_real = self.lossm(self.real_output, self.real_labels_patch)

        # self.fake_output = self.netD(self.fake_SB)
        self.fake_output = self.netD(self.fake_SB.detach())
        if len(self.fake_labels_patch) > len(self.fake_output):
            self.loss_D_fake = self.lossm(self.fake_output, self.fake_labels_patch[:len(self.fake_output)])
        else:
            self.loss_D_fake = self.lossm(self.fake_output, self.fake_labels_patch)
        self.loss_D = self.loss_D_real + self.loss_D_fake
        self.loss_D.backward()
        self.D_loss_sum_patch += self.loss_D.item()


    # def compute_visuals(self):
    #     # self.fake_TB = mul_mask(self.fake_TB, self.T_mask)
    #     self.fake_SB = self.fake_SB * 2 - 1


    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

