import torch
from models.base_model import BaseModel
from models import networks
from models.guided_filter_pytorch.guided_filter import FastGuidedFilter
from models.guided_filter_pytorch.sobel_filter import ThreeSobelFilter
from data.base_dataset import TensorToGrayTensor

class PixDAguideModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_DD', type=float, default=1, help='weight for DD')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for G loss')
            parser.add_argument('--lambda_DV', type=float, default=1, help='weight for DV loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.input_nc = opt.input_nc
        # self.opt.output_nc = 6 if 'guide' in opt.dataset_mode else opt.output_nc
        self.using_guide_filter = True if 'guide' in opt.dataset_mode else False
        self.guide_loss_lambda = opt.guide_loss_lambda
        # TODO：直接把整个文件作为一个模型
        # self.using_guide_loss = True
        self.using_guide_loss = True if 'guide' in opt.dataset_mode else False
        self.is_vector_D = True if 'vector' in opt.netG else False

        self.loss_names = ['DP', 'DP_fake', 'DP_real',
                           'DD', 'DD_fake_SB', 'DD_fake_TB',
                           'G', 'G_DP', 'G_L1', 'G_DD']
        self.visual_names = ['real_SA', 'fake_SB', 'real_SB', 'real_TA', 'fake_TB']

        if self.using_guide_loss:
            self.loss_names += ['G_guide']
            self.visual_names = ['real_SA', 'real_SAG', 'fake_SB', 'fake_SBG', 'real_SB', 'real_SBG', 'real_TA', 'real_TAG', 'fake_TB', 'fake_TBG']
        elif not self.using_guide_loss and self.using_guide_filter:
            self.visual_names = ['real_SA', 'real_SAG', 'fake_SB', 'real_SB', 'real_TA', 'real_TAG', 'fake_TB']

        # 初始化guide filter和灰度图工具
        if opt.edge_filter == 'guide_filter':
            self.edge_filter = FastGuidedFilter(self.device)
        else:
            self.edge_filter = ThreeSobelFilter(self.device)
        self.tensor_to_gray_tensor = TensorToGrayTensor(self.device)

        if self.isTrain:
            self.model_names = ['G', 'DD', 'DP']
            # 使用隐层判别器
            if self.is_vector_D:
                self.model_names += ['DV']
                self.loss_names += ['DV', 'G_DV']
        else:  # during test_total time, only load G
            self.model_names = ['G']

        # 网络的输出是3个channel
        self.netG = networks.define_G(opt.input_nc, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:
            # self.netDP = networks.define_D(3 + opt.output_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netDD = networks.define_D(3, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # 送进去
            if self.using_guide_loss:
                self.netDP = networks.define_D(opt.input_nc * 2, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netDP = networks.define_D(opt.input_nc * 2, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DP = torch.optim.Adam(self.netDP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DD = torch.optim.Adam(self.netDD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_DP)
            self.optimizers.append(self.optimizer_DD)

            # 使用隐层判别器
            if self.is_vector_D:
                self.netDV = networks.define_D(3, opt.ndfDV, 'Conv',
                                           norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
                self.optimizer_DV = torch.optim.Adam(self.netDV.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_DV)


    def set_input(self, input, isTrain=None):
        """
        处理输入
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
        self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
        self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
        if self.using_guide_filter:
            self.real_SAG = self.tensor_to_gray_tensor(self.real_SA)
            self.real_TAG = self.tensor_to_gray_tensor(self.real_TA)
            self.real_SAG = self.edge_filter(self.real_SAG)
            self.real_TAG = self.edge_filter(self.real_TAG)
            self.real_SA6 = torch.cat([self.real_SA, self.real_SAG], dim=1)
            self.real_TA6 = torch.cat([self.real_TA, self.real_TAG], dim=1)
            # 使用guide loss时才使用real_SBG
            if self.using_guide_loss:
                self.real_SBG = self.tensor_to_gray_tensor(self.real_SB)
                self.real_SBG = self.edge_filter(self.real_SBG)
                self.real_SB6 = torch.cat([self.real_SB, self.real_SBG], dim=1)
        self.image_paths = input['TA_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        if self.using_guide_filter and self.is_vector_D:
            self.fake_SB, self.SV = self.netG(self.real_SA6)  # G(SA)
            self.fake_TB, self.TV = self.netG(self.real_TA6)  # G(TA)
            self.fake_SBG = self.edge_filter(self.tensor_to_gray_tensor(self.fake_SB))
            self.fake_TBG = self.edge_filter(self.tensor_to_gray_tensor(self.fake_TB))
            self.fake_SB6 = torch.cat([self.fake_SB, self.fake_SBG], dim=1)
            self.fake_TB6 = torch.cat([self.fake_TB, self.fake_TBG], dim=1)
        elif self.using_guide_filter and not self.is_vector_D:
            self.fake_SB = self.netG(self.real_SA6)  # G(SA)
            self.fake_TB = self.netG(self.real_TA6)  # G(TA)
            self.fake_SBG = self.edge_filter(self.tensor_to_gray_tensor(self.fake_SB))
            self.fake_TBG = self.edge_filter(self.tensor_to_gray_tensor(self.fake_TB))
            self.fake_SB6 = torch.cat([self.fake_SB, self.fake_SBG], dim=1)
            self.fake_TB6 = torch.cat([self.fake_TB, self.fake_TBG], dim=1)
        elif self.is_vector_D:
            self.fake_SB, self.SV = self.netG(self.real_SA)  # G(SA)
            self.fake_TB, self.TV = self.netG(self.real_TA)  # G(TA)
        else:
            self.fake_SB = self.netG(self.real_SA)  # G(SA)
            self.fake_TB = self.netG(self.real_TA)  # G(TA)


    def backward_DD(self):
        """
        Calculate Domain loss for the discriminator, we want to discriminate S and T
        """
        if self.using_guide_filter:
            pred_fake_SB = self.netDD(self.fake_SB6.detach())
            self.loss_DD_fake_SB = self.criterionGAN(pred_fake_SB, True)
            # Fake Target, detach
            pred_fake_TB = self.netDD(self.fake_TB6.detach())
            self.loss_DD_fake_TB = self.criterionGAN(pred_fake_TB, False)
        # elif self.using_guide_filter and :
        #     # Fake Source, detach
        #     pred_fake_SB = self.netDD(torch.cat([self.fake_SB.detach(), self.fake_SBG.detach()], dim=1))
        #     self.loss_DD_fake_SB = self.criterionGAN(pred_fake_SB, True)
        #     # Fake Target, detach
        #     pred_fake_TB = self.netDD(torch.cat([self.fake_TB.detach(), self.fake_TBG.detach()], dim=1))
        #     self.loss_DD_fake_TB = self.criterionGAN(pred_fake_TB, False)
        else:
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

    # TODO：是否将patch对送给DP？
    def backward_DP(self):
        """
        Calculate GAN loss for the discriminator
        """
        if self.using_guide_filter:
            fake_SAB = torch.cat((self.real_SA6, self.fake_SB6), dim=1)
            pred_fake_SAB = self.netDP(fake_SAB.detach())
            self.loss_DP_fake = self.criterionGAN(pred_fake_SAB, False)

            real_SAB = torch.cat((self.real_SA6, self.real_SB6), dim=1)
            pred_real_SAB = self.netDP(real_SAB.detach())
            self.loss_DP_real = self.criterionGAN(pred_real_SAB, True)
        else:
            fake_SAB = torch.cat((self.real_SA, self.fake_SB), dim=1)
            pred_fake_SAB = self.netDP(fake_SAB.detach())
            self.loss_DP_fake = self.criterionGAN(pred_fake_SAB, False)
            # Real
            real_SAB = torch.cat((self.real_SA, self.real_SB), dim=1)
            pred_real_SAB = self.netDP(real_SAB.detach())
            self.loss_DP_real = self.criterionGAN(pred_real_SAB, True)

        # combine loss and calculate gradients
        # TODO: out的rate
        self.loss_DP = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_DP.backward()

    def backward_DV(self):
        pred_SV = self.netDV(self.SV.detach())
        self.loss_DV_SV = self.criterionGAN(pred_SV, True)
        pred_TV = self.netDV(self.TV.detach())
        self.loss_DV_TV = self.criterionGAN(pred_TV, False)
        self.loss_DV = (self.loss_DV_SV + self.loss_DV_TV) * 0.5 * self.opt.lambda_DV
        self.loss_DV.backward()

    def backward_G(self):
        """
        Calculate GAN and L1 loss for the generator
        Generator should fool the DD and DP
        """
        if self.using_guide_filter:
            # First, G(A) should fake the discriminator
            fake_SAB = torch.cat([self.real_SA6, self.fake_SB6], dim=1)
            pred_fake_SAB = self.netDP(fake_SAB)
            self.loss_G_DP = self.criterionGAN(pred_fake_SAB, True)

            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1

            # Third,  G(SA) and G(TA) should fool the domain discriminator
            # IMPORTANT
            pred_fake_SB6 = self.netDD(self.fake_SB6)
            pred_fake_TB6 = self.netDD(self.fake_TB6)
            self.loss_G_DD = (self.criterionGAN(pred_fake_TB6, True) + self.criterionGAN(pred_fake_SB6, False)) * \
                             0.5 * self.opt.lambda_DD
            # pred_fake_TB = self.netDD(self.fake_TB6)
            # self.loss_G_DD = self.criterionGAN(pred_fake_TB, True)
        else:
            # First, G(A) should fake the discriminator
            fake_SAB = torch.cat([self.real_SA, self.fake_SB], dim=1)
            pred_fake_SAB = self.netDP(fake_SAB)
            self.loss_G_DP = self.criterionGAN(pred_fake_SAB, True)

            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1

            # Third,  G(SA) and G(TA) should fool the domain discriminator
            # IMPORTANT
            pred_fake_SB = self.netDD(self.fake_SB)
            pred_fake_TB = self.netDD(self.fake_TB)
            self.loss_G_DD = (self.criterionGAN(pred_fake_TB, True) + self.criterionGAN(pred_fake_SB, False)) * \
                             self.opt.lambda_DD

        # 隐层loss
        if self.is_vector_D:
            pred_TV = self.netDV(self.TV)
            self.loss_G_DV = self.criterionGAN(pred_TV, True) * self.opt.lambda_DV
            # TODO: out的rate
            self.loss_G = self.loss_G_DP + self.loss_G_L1 + self.loss_G_DD + self.loss_G_DV
        else:
            self.loss_G = self.loss_G_DP + self.loss_G_L1 + self.loss_G_DD

        # guide loss
        if self.using_guide_loss:
            self.loss_G_guide = self.criterionL1(self.fake_SBG, self.real_SBG) * self.opt.guide_loss_lambda
            # self.loss_G_guide = 0
            self.loss_G += self.loss_G_guide

        self.loss_G = self.loss_G * self.opt.lambda_G
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
        # update DV(pixel)
        if self.is_vector_D:
            self.set_requires_grad(self.netDV, True)  # enable backprop for D
            self.optimizer_DV.zero_grad()  # set D's gradients to zero
            self.backward_DV()  # calculate gradients for D
            self.optimizer_DV.step()  # update D's weights
            self.set_requires_grad(self.netDV, False)  # D requires no gradients when optimizing G
        # update G
        self.set_requires_grad(self.netDD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netDP, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
