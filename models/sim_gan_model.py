import torch
from .base_model import BaseModel
from . import networks


class SimGANModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', source_size_count=1)

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_G_BV_OD', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for G loss')
            parser.add_argument('--lambda_G_D', type=float, default=0.1, help='weight for G loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test_total scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_BV_OD', 'G', 'D']
        # specify the images you want to save/display. The training/test_total scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_SA', 'fake_SB', 'real_SU', 'segmentation_mask']
        # specify the models you want to save to the disk. The training/test_total scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test_total time, only load G
            self.model_names = ['G']
            self.visual_names = ['real_SA', 'fake_SB', 'real_SB']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
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

    def set_input(self, input, isTrain=None):
        """Unpack input images from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the images itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
        self.segmentation_mask = input['segmentation_mask'].to(self.device)
        self.cup_mask = input['cup_mask'].to(self.device)
        self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
        self.real_SU = input['SU'].to(self.device)
        self.image_paths = input['SA_path' if AtoB else 'SB_path']
        if not self.isTrain or isTrain is not None:
            # self.real_SA = input['TB' if AtoB else 'TA'].to(self.device)
            self.image_paths = input['SA_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test_total>."""
        self.fake_SB = self.netG(self.real_SA)  # G(A)
        self.fake_SB = (self.fake_SB + 1) * self.cup_mask - 1

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_SB.detach())
        # D希望假就是假，真就是真
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        # pred_real是对真实图像对的预测分数
        pred_real = self.netD(self.real_SU.detach())
        # D希望假就是假，真就是真
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # D希望可以提高fake的loss，降低real的loss，但是在此处都是梯度下降
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_G_D
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_SB)
        # G希望pred_fake作为真，然后降低损失函数
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_G_D
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SA) * self.opt.lambda_L1
        self.loss_G_BV_OD = self.criterionL1((self.fake_SB+1) * self.segmentation_mask, (self.real_SA+1) * self.segmentation_mask) * self.opt.lambda_G_BV_OD
        # combine loss and calculate gradients
        # TODO:更正这个权重？
        self.loss_G = (self.loss_G_GAN + self.loss_G_L1 + self.loss_G_BV_OD) * self.opt.lambda_G
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
