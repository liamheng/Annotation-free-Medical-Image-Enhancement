"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input_color-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input_color data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test_cat>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import os

import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from colorspacious import cspace_convert
from PIL import Image, ImageCms
import torchvision.transforms as transforms
from models.loss.MSL import MSL
from models.loss.entro import entropy

from util.EMA import EMA
import matplotlib.pyplot as plt
class SDAKDgradChallengeSobelModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test_cat phase. You can use this flag to add training-specific or test_cat-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned data as its data.
        if is_train:
            parser.add_argument('--lambda_l1', type=float, default=0.5)
            parser.add_argument('--lambda_cl', type=float, default=0.5)
            parser.add_argument('--lambda_ent', type=float, default=1)
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test_cat options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # TODO : MSL
        self.loss_names = ['G_L1','G_seg','entro']
        # self.visual_names = ['real_EyeQ', 'fake_EyeQ', 'real_seg', 'fake_seg']
        # self.loss_names = ['G_L1']

        # TODO : SDA
        # self.visual_names_train = ['gt', 's_fake','target_noise',
        #                            't_fake','s_fake_seg','t_fake_seg','real_seg','noise_seg']
        # self.visual_names_train = ['gt', 's_fake','target_noise',
        #                            't_fake','s_fake_seg','t_fake_seg','real_seg']
        self.visual_names_train = ['target_gt', 's_fake','target_noise',
                                   't_fake','s_fake_seg','t_fake_seg']
        # self.visual_names_test = ['target_gt', 's_fake','target_noise','t_fake','s_fake_seg','t_fake_seg']
        self.visual_names_test = ['s_fake']
        # 它返回一个普通的cmsprofile对象，可以将该对象传递给imagecms.buildTransformFromOpenProfiles（），以创建一个应用于图像的转换。


        if self.isTrain:
            self.model_names = ['S','T']
            self.visual_names = self.visual_names_train
        else:
            self.model_names = ['S','T']
            self.visual_names = self.visual_names_test
        # define networks; you can use opt.isTrain to specify different behaviors for training and test_cat.
        # self.netS = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        # self.netT = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        self.netS = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netT = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # 初始化ema
        self.ema = EMA(mu=0.995)
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            # TODO
            self.criterionContent = torch.nn.L1Loss()
            self.criterionSegment = torch.nn.L1Loss()
            # self.criterionSegment = torch.nn.L1Loss()
            # self.msl = MSL()

            self.entro = entropy()

            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            # 梯度只反传到Student网络
            self.optimizer = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input, isTrain=None,model=None):
        """Unpack input_color data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        # self.data_A = input_color['A' if AtoB else 'B'].to(self.device)  # get image data A
        # self.data_B = input_color['B' if AtoB else 'A'].to(self.device)  # get image data B
        # self.image_paths = input_color['TA_path' if AtoB else 'SA_path']  # get image paths


        if isTrain :
            self.image_paths = input['SA_path']  # get image paths
            self.target_noise = input['source_noise'].to(self.device)
            self.target_gt = input['source_gt'].to(self.device)


            # TODO
            # self.noise_seg = input_color['noise_seg'].to(self.device)
            # 获取mcf
            self.mcf = model
        else:
            self.image_paths = input['SA_path']  # get image paths
            self.target_noise = input['source_noise'].to(self.device)
            self.target_gt = input['source_gt'].to(self.device)


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test_cat>."""
        self.s_fake, self.s_fake_seg = self.netS(self.target_noise)  # generate output image given the input_color data_A
        self.t_fake, self.t_fake_seg = self.netT(self.target_noise)

        if self.isTrain:
            quality = self.mcf(self.s_fake)

            self.s_pesudo = torch.FloatTensor().cuda()
            self.t_pesudo = torch.FloatTensor().cuda()
            self.s_seg_pesudo = torch.FloatTensor().cuda()
            self.t_seg_pesudo = torch.FloatTensor().cuda()

            for i in range(quality.shape[0]):
                # 标签为接受并且概率大于0.8
               if (quality[i][0] > 0.6) & (torch.argmax(quality[i]) == 0):
                    self.s_pesudo = torch.cat((self.s_pesudo, self.s_fake[i].unsqueeze(0)), 0) # unsqueeze(0)增加一个维度
                    self.t_pesudo = torch.cat((self.t_pesudo, self.t_fake[i].unsqueeze(0)), 0)
                    self.s_seg_pesudo = torch.cat((self.s_seg_pesudo, self.s_fake_seg[i].unsqueeze(0)), 0)
                    self.t_seg_pesudo = torch.cat((self.t_seg_pesudo, self.t_fake_seg[i].unsqueeze(0)), 0)



    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input_color and intermediate results


        self.loss_G_L1 = self.criterionContent(self.s_pesudo, self.t_pesudo) * self.opt.lambda_l1
        self.loss_G_seg = self.criterionSegment(self.s_seg_pesudo, self.t_seg_pesudo) * self.opt.lambda_cl
        # TODO ：系数
        # self.loss_MSL = self.msl(self.s_fake_seg)
        # self.loss_entro = self.entro(self.s_fake_seg) * self.opt.lambda_ent
        self.loss_entro = self.entro(self.s_fake_seg) * self.opt.lambda_ent
        self.loss_G = self.loss_G_L1 + self.loss_G_seg + self.loss_entro
        # self.loss_G = self.loss_G_L1 + self.loss_G_seg

        self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G
        # else:
        #     self.loss_G_L1 = self.criterionContent(self.s_pesudo, self.t_pesudo) * self.opt.lambda_l1
        #     self.loss_G_seg = self.criterionSegment(self.s_seg_pesudo, self.t_seg_pesudo) * self.opt.lambda_cl
        #     # TODO ：系数
        #     # self.loss_MSL = self.msl(self.s_fake_seg)
        #     # self.loss_entro = self.entro(self.s_fake_seg) * self.opt.lambda_ent
        #     self.loss_entro = self.entro(self.s_seg_pesudo) * self.opt.lambda_ent
        #     self.loss_G = self.loss_G_L1 + self.loss_G_seg + self.loss_entro
        #     # self.loss_G = self.loss_G_L1 + self.loss_G_seg
        #
        #     self.loss_G.backward()  # calculate gradients of network G w.r.t. loss_G


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
        # 获取学生模型权重
        for name, param in self.netS.named_parameters():
            self.ema.register(name, param)
        # 更新
        for name, param in self.netT.named_parameters():
            param.data = self.ema(name,param.data)
        # TODO : 可视化
        self.compute_visuals()
    def compute_visuals(self):
        # TODO 修改可视化
        pass
        # self.segmentation_mask[self.segmentation_mask >= 0.5] = 1
        # self.segmentation_mask[self.segmentation_mask < 0.5] = -1
        # self.s_fake_seg[self.s_fake_seg >= 0.5] = 1
        # self.s_fake_seg[self.s_fake_seg < 0.5] = -1
        # self.t_fake_seg[self.t_fake_seg >= 0.5] = 1
        # self.t_fake_seg[self.t_fake_seg < 0.5] = -1
        # self.real_seg[self.real_seg >= 0.5] = 1
        # self.real_seg[self.real_seg < 0.5] = -1
    # def compute_visuals(self):
    #     self.fake_artifact = torch.nn.functional.interpolate(self.fake_artifact, size=(256, 256), mode='nearest')

    def train(self):
        """Make models eval mode during test_total time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # 训练和测试阶段加载的初始化不一样
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            self.load_networks2()

        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def load_networks2(self):

        """Load all the networks from the disk.

        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = 'latest_net_%s.pth' % (name)
                # TODO : 修改路径
                load_path = os.path.join("./pre_trained_model/stillgan_scr_cyclesobel_lr0.001", load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

