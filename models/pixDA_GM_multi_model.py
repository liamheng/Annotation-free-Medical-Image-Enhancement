import torch
import itertools
from models.guided_filter_pytorch.HFC_filter import HFCFilter
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.gaussian_filter2 import OneGaussianFilter
from data.base_dataset import TensorToGrayTensor

class PixDAGMMultiModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L1G', type=float, default=50.0, help='weight for L1G loss')
            parser.add_argument('--lambda_DDP', type=float, default=0, help='weight for DDP')
            parser.add_argument('--lambda_DP', type=float, default=0, help='weight for G loss')

            parser.add_argument('--RMS', action='store_true',)
        # parser.add_argument('--filter_width', type=int, default=25, help='weight for G loss')
        # parser.add_argument('--nsig', type=int, default=20, help='weight for G loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.input_nc = opt.input_nc

        self.loss_names = ['DP', 'DP_fake', 'DP_real',
                           'DDP', 'DDP_fake_SB', 'DDP_fake_TB',
                           'G', 'G_DP', 'G_L1', 'G_L1G', 'G_DDP']

        self.visual_names = ['real_SA', 'fake_SB', 'fake_SB_G',
                             'real_SB', 'real_SB_G',
                             'real_TA',
                             'fake_TB', 'fake_TB_G']
        if self.isTrain:
            self.model_names = ['G', 'DDP', 'DP']
        else:  # during test_total time, only load G
            self.model_names = ['G']
            # self.visual_names = ['real_TA', 'real_TAG',
            #                      'fake_TB', 'fake_TBG']
            self.visual_names = ['fake_TB']

        # 网络的输出是3个channel
        self.netG = networks.define_G(opt.input_nc, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:
            self.netDP = networks.define_D(3, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDDP = networks.define_D(3, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.hfc_filter_x = HFCFilter(21, 20, sub_mask=True).to(self.device)
            self.avg_pool = torch.nn.AvgPool2d(2).to(self.device)

            # optimizers
            # 增加使用RMS
            if not self.opt.RMS:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netDP.parameters(),
                                                                    self.netDDP.parameters()), lr=opt.lr,
                                                    betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr, alpha=0.9)
                self.optimizer_D = torch.optim.RMSprop(itertools.chain(self.netDP.parameters(),
                                                                    self.netDDP.parameters()), lr=opt.lr,
                                                    alpha=0.9)

            self.optimizers.append(self.optimizer_G)
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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        self.fake_SB, self.fake_SB_1, self.fake_SB_2, self.fake_SB_3 = self.netG(self.real_SA, self.S_mask)  # G(SA)
        self.fake_TB, self.fake_TB_1, self.fake_TB_2, self.fake_TB_3 = self.netG(self.real_TA, self.T_mask)  # G(SA)
        self.fake_SB_G = self.hfc_filter_x(self.fake_SB, self.S_mask)
        self.real_SB_G = self.hfc_filter_x(self.real_SB, self.S_mask)
        self.fake_TB_G = self.hfc_filter_x(self.fake_TB, self.T_mask)
        self.real_SB_1 = self.avg_pool(self.real_SB)
        self.real_SB_2 = self.avg_pool(self.real_SB_1)
        self.real_SB_3 = self.avg_pool(self.real_SB_2)

        if not self.isTrain:
            self.fake_TB = (self.fake_TB + 1) * self.T_mask - 1

        # self.fake_SAB = torch.cat((self.real_SA, self.fake_SB), dim=1)
        # self.real_SAB = torch.cat((self.real_SA, self.real_SB), dim=1)


    def backward_DDP(self):
        """
        Calculate Domain loss for the discriminator, we want to discriminate S and T
        """
        # Fake Target, detach
        pred_fake_SB = self.netDDP(self.fake_SB.detach())
        pred_fake_TB = self.netDDP(self.fake_TB.detach())

        self.loss_DDP_fake_SB = self.criterionGAN(pred_fake_SB, True) * self.opt.lambda_DDP
        self.loss_DDP_fake_TB = self.criterionGAN(pred_fake_TB, False) * self.opt.lambda_DDP

        # combine loss and calculate gradients
        self.loss_DDP = (self.loss_DDP_fake_SB + self.loss_DDP_fake_TB) * 0.5
        self.loss_DDP.backward()

    # TODO：是否将patch对送给DP？
    def backward_DP(self):
        """
        Calculate GAN loss for the discriminator
        """
        pred_fake_SB = self.netDP(self.fake_SB.detach())
        pred_real_SB = self.netDP(self.real_SB.detach())

        self.loss_DP_fake = self.criterionGAN(pred_fake_SB, False)
        self.loss_DP_real = self.criterionGAN(pred_real_SB, True)

        # combine loss and calculate gradients
        self.loss_DP = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_DP.backward()


    def backward_G(self):
        """
        Calculate GAN and L1 loss for the generator
        Generator should fool the DD and DP
        """
        # First, G(A) should fake the discriminator
        pred_fake_SB = self.netDP(self.fake_SB)

        self.loss_G_DP = self.opt.lambda_DP * self.criterionGAN(pred_fake_SB, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1
        self.loss_G_L1 += self.criterionL1(self.fake_SB_1, self.real_SB_1) * self.opt.lambda_L1 / 2
        self.loss_G_L1 += self.criterionL1(self.fake_SB_2, self.real_SB_2) * self.opt.lambda_L1 / 3
        self.loss_G_L1 += self.criterionL1(self.fake_SB_3, self.real_SB_3) * self.opt.lambda_L1 / 4

        self.loss_G_L1G = self.criterionL1(self.fake_SB_G, self.real_SB_G) * self.opt.lambda_L1G




        # Third,  G(SA) and G(TA) should fool the domain discriminator
        pred_fake_TB = self.netDDP(self.fake_TB)
        pred_fake_SB = self.netDDP(self.fake_SB)
        self.loss_G_DDP = (self.criterionGAN(pred_fake_TB, True) + self.criterionGAN(pred_fake_SB, False)) * \
                         0.5 * self.opt.lambda_DDP

        self.loss_G = self.loss_G_DP + self.loss_G_DDP + self.loss_G_L1 + self.loss_G_L1G
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update DD (domain discriminator)
        self.set_requires_grad([self.netDP, self.netDDP], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_DP()      # calculate gradients for D_A
        self.backward_DDP()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # update G
        self.set_requires_grad([self.netDP, self.netDDP],
                               False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

