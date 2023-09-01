import torch
from models.base_model import BaseModel
from models import networks


class Pix2PixDehazeNoFeatureModel(BaseModel):
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
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_G_DP', type=float, default=1, help='weight for G loss')
            parser.add_argument('--lambda_G_BV_OD', type=float, default=100.0, help='weight for L1 loss')
            # parser.add_argument('--lambda_G_multi', type=float, default=1000.0, help='weight for G loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_BV_OD', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test_total scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'S_mask']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test_total time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.w_ratio = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]

    def set_input(self, input, isTrain=None):
        """Unpack input images from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the images itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['SA' if AtoB else 'SB'].to(self.device)
        self.real_B = input['SB' if AtoB else 'SA'].to(self.device)
        self.S_mask = input['S_mask'].to(self.device)
        self.image_paths = input['SA_path' if AtoB else 'SB_path']
        if not self.isTrain or isTrain is not None:
            self.real_A = input['TA' if AtoB else 'TB'].to(self.device)
            self.image_paths = input['TA_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        # self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # real_A和fake_B的维度为[1,3,256,256]
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # 梯度不再往fake_AB前传
        # pred_fake是对于虚假图像对的预测分数
        pred_fake = self.netD(fake_AB.detach())
        # D希望假就是假，真就是真
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        # pred_real是对真实图像对的预测分数
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        # D希望假就是假，真就是真
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        # D希望可以提高fake的loss，降低real的loss，但是在此处都是梯度下降
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_G_DP
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # G希望pred_fake变小
        pred_fake = self.netD(fake_AB)
        # G希望pred_fake作为真，然后降低损失函数
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_G_DP
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G_BV_OD = self.criterionL1(self.fake_B * self.S_mask,
                                             self.real_B * self.S_mask) * self.opt.lambda_G_BV_OD
        # self.loss_G_multi = 0
        # for (SA_f, SB_f, r) in zip(self.real_A_feature, self.fake_B_feature, self.w_ratio):
        #     self.loss_G_multi += r * self.criterionL1(SA_f, SB_f)
        # self.loss_G_multi = self.opt.lambda_G_multi * self.loss_G_multi
        # self
        # TODO:更正这个权重？
        self.loss_G = self.loss_G_GAN + self.loss_G_BV_OD + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
