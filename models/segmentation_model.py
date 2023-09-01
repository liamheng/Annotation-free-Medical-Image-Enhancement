# -*- coding: UTF-8 -*-
"""
@Function:
@File: segmentation_model.py
@Date: 2021/7/13 19:55 
@Author: Hever
"""
import torch
from models.backbone.unet_open_source import UNet
from .base_model import BaseModel
# from . import networks
from util import util
from skimage.metrics import mean_squared_error, normalized_root_mse

def define_seg_model(input_nc, output_nc, gpu_ids=[]):
    net = UNet(input_nc, output_nc)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net


class SegmentationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(netG='Unet', dataset_mode='aligned', load_size=256, crop_size=256, output_nc=1)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_CE', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--mask_the_ves', action='store_true')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G', 'CE', 'L1']
        self.visual_names = ['real_A', 'real_B', 'fake_B']
        self.model_names = ['Seg']
        self.netSeg = define_seg_model(opt.input_nc, opt.output_nc, self.gpu_ids)
        if self.isTrain:  # define discriminators
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.l1_loss = torch.nn.L1Loss()
            self.optimizer_Seg = torch.optim.Adam(self.netSeg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Seg)
        self.loss_L1 = 0
        self.threshold_list = [0.15686*0.5, 0.15686*1.5, 0.15686*2.5, 0.15686*3.5, 0.15686*4.5, 0.15686*5.5, 1]


    def set_input(self, input, isTrain=None):
        # if isTrain:
        self.real_A = input['images'].to(self.device)
        # self.mask = input['mask'].to(self.device)
        self.real_B = input['ves'].to(self.device)
        self.image_paths = input['image_path']
        # else:
        #     self.real_A = input['images'].to(self.device)
        #     self.mask = input['mask'].to(self.device)
        #     self.image_paths = input['image_path']

    def forward(self):
        self.fake_B = self.netSeg(self.real_A)  # G(A)
        pass

    def compute_visuals(self):
        # TODO 修改可视化
        # self.fake_B[self.fake_B >= 0.5] = 1
        # self.fake_B[self.fake_B < 0.5] = -1
        # self.fake_B torch.Size([8, 7, 256, 256])
        # TODO ：7分类取max  cls是对于的pesudo_label
        cls = torch.argmax(self.fake_B, axis=1).unsqueeze(1).to(torch.float32) #
        #  还原成对于的像素点
        self.fake_B = cls * 0.15686 * 2 - 1
        gt = self.real_B.unsqueeze(1).to(torch.float32)
        self.real_B = gt * 0.15686 * 2 - 1
        # print(self.fake_B.shape, self.real_B.shape)
        # pass
        # self.fake_B[self.fake_B <= 0.1] = -1
        # self.real_B[self.real_B > 0.5] = 1
        # self.real_B[self.real_B <= 0.5] = -1
        # self.real_B[self.real_B <= 0.1] = -1

        # if self.opt.mask_the_ves:
        #     self.fake_B = self.fake_B * self.mask

    def test(self):
        with torch.no_grad():
            self.fake_B = self.netSeg(self.real_A)  # G(A)
            self.compute_visuals()
            # self.fake_B[self.fake_B >= 0.5] = 1
            # self.fake_B[self.fake_B < 0.5] = -1

            # if self.opt.mask_the_ves:
            #     self.fake_B = self.fake_B * self.mask

            # if self.opt.eval_when_train:
            #     self.eval_when_train()

    def eval_when_train(self):
        pass
        # self.ssim = self.psnr = 0
        # self.real_B[self.real_B > 0.5] = 1
        # self.real_B[self.real_B <= 0.5] = -1
        # for i in range(len(self.fake_B)):
        #     self.fake_B_im = util.tensor2im(self.fake_B[i:i+1])
        #     self.real_B_im = util.tensor2im(self.real_B[i:i+1])
        #     self.ssim += mean_squared_error(self.real_B_im, self.fake_B_im)
        #     self.psnr += normalized_root_mse(self.real_B_im, self.fake_B_im)

    def backward_Seg(self):
        self.loss_CE = self.criterionCE(self.fake_B, self.real_B) * self.opt.lambda_CE
        self.loss_G = self.loss_CE
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        self.optimizer_Seg.zero_grad()     # set D's gradients to zero
        self.backward_Seg()                # calculate gradients for D
        self.optimizer_Seg.step()          # update D's weights
        # self.compute_visuals()

