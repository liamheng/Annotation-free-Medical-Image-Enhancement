import torch
import itertools
from models.base_model import BaseModel
from models import networks
from models.guided_filter_pytorch.gaussian_filter import OneGaussianFilter
from data.base_dataset import TensorToGrayTensor
from models.backbone.RSA_module import RSAModule



def define_seg_model(input_nc, output_nc, device, gpu_ids=[]):
    net = RSAModule(input_nc, output_nc, resnet_pretrain=False, get_feature=False)
    net.load_state_dict(torch.load('./pre_trained_model/net_RSA.pth', map_location=str(device)))
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net


class PixDADDPDP3SegModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_CE', type=float, default=30.0, help='weight for L1 loss')
            parser.add_argument('--lambda_DDP', type=float, default=1, help='weight for DDP')
            parser.add_argument('--lambda_DP', type=float, default=1, help='weight for G loss')

            parser.add_argument('--RMS', action='store_true',)

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
                           'G', 'G_DP', 'G_L1', 'G_CE', 'G_DDP']

        self.visual_names = ['real_SA', 'real_SAS', 'fake_SB',
                             'fake_SBS', 'real_SB', 'real_SBS',
                             'real_TA', 'real_TAS',
                             'fake_TB', 'fake_TBS']
        # 初始化guide filter和灰度图工具

        if self.isTrain:
            self.model_names = ['G', 'DDP', 'DP', 'Seg']
        else:  # during test_total time, only load G
            self.model_names = ['G', 'Seg']

        # 网络的输出是3个channel
        self.netG = networks.define_G(opt.input_nc, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netSeg = define_seg_model(3, 1, self.device, self.gpu_ids)
        self.set_requires_grad(self.netSeg, False)  # enable backprop for D


        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:
            self.netDP = networks.define_D(3, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDDP = networks.define_D(3, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCE = torch.nn.BCELoss()

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

        self.real_SBS = self.netSeg(self.real_SB)

        self.image_paths = input['TA_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        self.real_SAS = self.netSeg(self.real_SA)
        self.real_TAS = self.netSeg(self.real_TA)

        self.real_SA4 = torch.cat([self.real_SA, self.real_SAS], dim=1)
        self.real_TA4 = torch.cat([self.real_TA, self.real_TAS], dim=1)

        self.fake_SB = self.netG(self.real_SA4)  # G(SA)
        self.fake_TB = self.netG(self.real_TA4)  # G(TA)

        self.fake_SBS = self.netSeg(self.fake_SB)
        self.fake_TBS = self.netSeg(self.fake_TB)


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
        self.loss_G_CE = self.criterionCE(self.fake_SBS, self.real_SAS) * self.opt.lambda_CE

        # Third,  G(SA) and G(TA) should fool the domain discriminator
        pred_fake_TB = self.netDDP(self.fake_TB)
        pred_fake_SB = self.netDDP(self.fake_SB)
        self.loss_G_DDP = (self.criterionGAN(pred_fake_TB, True) + self.criterionGAN(pred_fake_SB, False)) * \
                         0.5 * self.opt.lambda_DDP

        self.loss_G = self.loss_G_DP + self.loss_G_DDP + self.loss_G_L1 + self.loss_G_CE
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

