import torch
import models.backbone.MPR_losses as losses
import numpy as np
from .networks import init_net
from .base_model import BaseModel
# from . import networks
from .backbone.MPRNet import MPRNet


class MPRModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for G loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['char', 'edge', 'G']
        # specify the images you want to save/display. The training/test_total scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_SB_1', 'fake_SB_2', 'fake_SB_3']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test_total time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = MPRNet(in_c=3, out_c=3)
        self.netG = init_net(self.netG, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterion_char = losses.CharbonnierLoss()
            self.criterion_edge = losses.EdgeLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input, isTrain=None):
        """Unpack input images from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the images itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['SA' if AtoB else 'SB'].to(self.device)
        self.real_B = input['SB' if AtoB else 'SA'].to(self.device)
        self.image_paths = input['SA_path' if AtoB else 'SB_path']
        if not self.isTrain or isTrain is not None:
            self.real_A = input['TA' if AtoB else 'TB'].to(self.device)
            self.image_paths = input['TA_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        self.restored = self.netG(self.real_A)  # G(A)
        self.fake_SB_1 = self.restored[0]
        self.fake_SB_2 = self.restored[1]
        self.fake_SB_3 = self.restored[2]

    def backward_G(self):
        # Compute loss at each stage
        self.loss_char = np.sum([self.criterion_char(self.restored[j], self.real_B) for j in range(len(self.restored))])
        self.loss_edge = np.sum([self.criterion_edge(self.restored[j], self.real_B) for j in range(len(self.restored))])
        self.loss_G = (self.loss_char) + (0.05 * self.loss_edge)
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
