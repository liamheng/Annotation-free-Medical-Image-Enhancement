import torch
from .base_model import BaseModel
from . import networks


class PixDAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_DD', type=float, default=1, help='weight for DD')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for G loss')
            parser.add_argument('--lambda_G_DD', type=float, default=1, help='weight for G loss')
            parser.add_argument('--lambda_G_DD_SB', type=float, default=1, help='weight for DD')



        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        # self.loss_D = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_names = ['DP_fake', 'DP_real',
                           'DD_fake_SB', 'DD_fake_TB', 'DD',
                           'G_DD_TB', 'G_DD_SB',
                           'G_DP', 'G_L1', 'G_DD']
        # 'G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test_total scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B']#, 'real_B']
        # self.visual_names = ['real_SA', 'fake_SB', 'real_SB', 'real_TA', 'fake_TB', 'real_TB']
        self.visual_names = ['real_SA', 'fake_SB', 'real_SB', 'real_TA', 'fake_TB']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'DD', 'DP']
        else:  # during test_total time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netDP = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            #self.netOut = networks.define_out(opt.output_nc, 256, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionOut = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DP = torch.optim.Adam(self.netDP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DD = torch.optim.Adam(self.netDD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_DP)
            self.optimizers.append(self.optimizer_DD)

    def set_input(self, input, isTrain=None):
        """
        处理输入
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
        self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
        self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
        # self.real_TB = input['TB' if AtoB else 'TA'].to(self.device)
        self.image_paths = input['TA_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        self.fake_SB = self.netG(self.real_SA)  # G(SA)
        self.fake_TB = self.netG(self.real_TA)  # G(TA)

    def backward_DD(self):
        """
        Calculate Domain loss for the discriminator, we want to discriminate S and T
        TODO:哪种判别器好？
        """
        # Fake Source, detach
        pred_fake_SB = self.netDD(self.fake_SB.detach())
        self.loss_DD_fake_SB = self.criterionGAN(pred_fake_SB, True)
        # Fake Target, detach
        pred_fake_TB = self.netDD(self.fake_TB.detach())
        self.loss_DD_fake_TB = self.criterionGAN(pred_fake_TB, False)
        # combine loss and calculate gradients
        # TODO: out的rate
        self.loss_DD = (self.loss_DD_fake_SB + self.loss_DD_fake_TB) * 0.5 * self.opt.lambda_DD
        self.loss_DD.backward()

    def backward_DP(self):
        """
        Calculate GAN loss for the discriminator
        """
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_SAB = torch.cat((self.real_SA, self.fake_SB), 1)
        pred_fake_SAB = self.netDP(fake_SAB.detach())
        self.loss_DP_fake = self.criterionGAN(pred_fake_SAB, False)
        # Real
        real_SAB = torch.cat((self.real_SA, self.real_SB), 1)
        pred_real_SAB = self.netDP(real_SAB)
        self.loss_DP_real = self.criterionGAN(pred_real_SAB, True)
        # combine loss and calculate gradients
        # TODO: out的rate
        self.loss_DP = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_DP.backward()

    def backward_G(self):
        """
        Calculate GAN and L1 loss for the generator
        Generator should fool the DD and DP
        """
        # First, G(A) should fake the discriminator
        fake_SAB = torch.cat((self.real_SA, self.fake_SB), 1)
        pred_fake_SAB = self.netDP(fake_SAB)
        self.loss_G_DP = self.criterionGAN(pred_fake_SAB, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1
        # Third,  G(SA) and G(TA) should fool the domain discriminator
        # IMPORTANT
        pred_fake_SB = self.netDD(self.fake_SB)
        pred_fake_TB = self.netDD(self.fake_TB)
        self.loss_G_DD_TB = self.criterionGAN(pred_fake_TB, True)
        self.loss_G_DD_SB = self.criterionGAN(pred_fake_SB, False) * self.opt.lambda_G_DD_SB

        self.loss_G_DD = (self.loss_G_DD_TB + self.loss_G_DD_SB) * 0.5 * self.opt.lambda_G_DD
        # TODO: out的rate
        self.loss_G = (self.loss_G_DP + self.loss_G_L1 + self.loss_G_DD) * self.opt.lambda_G
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update DD (domain discriminator)
        self.set_requires_grad(self.netDP, True)  # enable backprop for Out
        self.optimizer_DP.zero_grad()     # set Out's gradients to zero
        self.backward_DP()                # calculate gradients for Out
        self.optimizer_DP.step()          # update Out's weights
        # update DP(pixel)
        self.set_requires_grad(self.netDD, True)  # enable backprop for D
        self.optimizer_DD.zero_grad()     # set D's gradients to zero
        self.backward_DD()                # calculate gradients for D
        self.optimizer_DD.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netDD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netDP, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
