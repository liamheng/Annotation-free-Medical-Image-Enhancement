import torch
from .base_model import BaseModel
from . import networks
from models.BSRI.MPRNet import MPRNet
from models.networks import init_weights
import models.BSRI.losses as losses
import numpy as np
path = './models/BSRI/AG_Net.pth'

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def load_pretrained_personal_model(model, path='./models/BSRI/AG_Net.pth', gpu_ids=[0]):
    pretrain_dict = torch.load(path,map_location='cpu')
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        k_ori = k[7:]
        if k_ori in state_dict:
            model_dict[k_ori] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = torch.nn.DataParallel(model, gpu_ids)  # multi-GPUs
    return model

def create_model(ema=False, gpus_ids=[0]):
    # Network definition

    if ema:
        net = MPRNet()
        net = load_pretrained_personal_model(net, path = path, gpu_ids=gpus_ids)
        # net_cuda = net.cuda()

        for param in net.parameters():
            param.detach_()
    else:
        net = MPRNet()
        net = load_pretrained_personal_model(net, path = path, gpu_ids=gpus_ids)
        # net_cuda = net.cuda()

    return net

def mul_mask(image, mask):
    return (image + 1) * mask - 1

class BSRIModel(BaseModel):
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
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_D = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_names = ['supervised', 'consistency', 'char', 'edge', 'ves', 'ema', 'ves_cons']

        self.visual_names_train = ['real_SA', 'real_SB', 'real_SB_ves', 'real_TA', 'ema_input',
                             'fake_SB_stage1', 'fake_SB_stage2',
                             'fake_TB_stage1', 'fake_TB_stage2',
                             'fake_TB_ema_stage1', 'fake_TB_ema_stage2',
                             'fake_TB_ema_ves_vis',
                             'fake_SB_ves_vis', 'fake_TB_ves_vis',
                             ]
        self.visual_names_test = ['real_TA', 'fake_TB']

        if self.isTrain:
            self.model_names = ['G', 'G_ema']
            self.visual_names = self.visual_names_train

        else:  # during test time, only load G
            self.model_names = ['G', 'G_ema']
            self.visual_names = self.visual_names_test

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = create_model(ema=False, gpus_ids=self.gpu_ids)
        self.netG_ema = create_model(ema=True, gpus_ids=self.gpu_ids)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc

            self.criterion_char = losses.CharbonnierLoss()
            self.criterion_edge = losses.EdgeLoss()
            self.segmen_loss = losses.dice_bce_loss()
            self.segmen_loss.initialize()
            self.consistency_criterion = losses.sigmoid_mse_loss
            # define loss functions
            self.criterion = torch.nn.NLLLoss2d()
            self.softmax_2d = torch.nn.Softmax(dim=1)
            self.bce_loss = torch.nn.BCELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G_ema = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.ce_loss = torch.nn.CrossEntropyLoss()
            self.CONSISTENCY = 1.0
            self.CONSISTENCY_RAMPUP = 100.0
            self.EMA_DECAY = 0.99

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.CONSISTENCY * self.sigmoid_rampup(epoch, self.CONSISTENCY_RAMPUP)

    def set_input(self, input, isTrain=None):
        AtoB = self.opt.direction == 'AtoB'
        if not self.isTrain or isTrain == False:
            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.image_paths = input['TA_path']
        else:
            self.real_SA = input['SA'].to(self.device)
            self.real_SB = input['SB'].to(self.device)
            self.real_SB_ves = input['SA_seg'].to(self.device)
            self.real_SB_ves_label = self.real_SB_ves.long()
            self.real_SB_ves_label = self.real_SB_ves_label.squeeze(1)
            self.real_TA = input['TA'].to(self.device)
            # self.real_TB = input['TB' if AtoB else 'TA'].to(self.device)
            self.image_paths = input['TA_path']
            # 有label的16个，无label的8个
            noise = torch.clamp(torch.randn_like(self.real_TA) * 0.1, -0.1, 0.1)
            self.ema_input = self.real_TA + noise
            self.ema_input = self.ema_input.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with torch.no_grad():
            self.fake_TB_ema, self.fake_TB_ema_ves = self.netG_ema(self.real_TA)  # G(TA)
        # self.fake_B, self.fake_B_ves = self.netG(torch.cat([self.real_SA, self.real_TA], dim=0))  # G(SA)
        self.fake_SB, self.fake_SB_ves = self.netG(self.real_SA)  # G(SA)
        self.fake_TB, self.fake_TB_ves = self.netG(self.real_TA)  # G(SA)

        # self.SA_batch_size = self.real_SA.shape[0]
        # self.TA_batch_size = self.real_TA.shape[0]
        # self.fake_SB, self.fake_SB_ves = self.fake_B[:self.SA_batch_size], self.fake_B_ves[:self.SA_batch_size]
        # self.fake_SB, self.fake_SB_ves = self.fake_B[self.SA_batch_size:], self.fake_B_ves[self.SA_batch_size:]

        # self.fake_TB, self.fake_TB_ves = self.netG(self.real_TA)  # G(TA)

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            self.fake_TB_ema, self.fake_TB_ema_ves = self.netG_ema(self.real_TA)
        self.fake_TB = self.fake_TB_ema[0]
        self.fake_TB = mul_mask(self.fake_TB, self.T_mask)

    def train(self):
        """Make models eval mode during test time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        loss_char = []
        loss_edge = []
        loss_ema = []
        for (ix, ix_ema) in zip(self.fake_SB, self.fake_TB_ema):
            #supervised
            # loss_char.append(self.criterion_char(torch.clamp(ix,-1,1), self.real_SB))
            # loss_edge.append(self.criterion_edge(torch.clamp(ix,-1,1), self.real_SB))
            # #consistency
            # loss_ema.append(self.consistency_criterion(torch.clamp(ix,-1,1), torch.clamp(ix_ema,-1,1)))
            loss_char.append(self.criterion_char(ix, self.real_SB))
            loss_edge.append(self.criterion_edge(ix, self.real_SB))
            # consistency
            loss_ema.append(self.consistency_criterion(ix, ix_ema))
        # self.loss_char = self.criterion_char(torch.clamp(self.fake_SB, 0, 1), self.real_SB)
        # self.loss_edge = self.criterion_edge(torch.clamp(self.fake_SB, 0, 1), self.real_SB)
        # self.loss_ema = self.consistency_criterion(torch.clamp(self.fake_TB, 0, 1), torch.clamp(self.fake_TB_ema, 0, 1))
        self.loss_char = sum(loss_char)
        self.loss_edge = sum(loss_edge)
        self.loss_ema = sum(loss_ema)

        self.loss_ves = 0
        # for i, j in zip(self.fake_SB_ves, self.real_SB_ves_label):
        #     # self.loss_ves += self.criterion(self.softmax_2d(i), self.softmax_2d(j))
        #     self.loss_ves += self.ce_loss(i, j)
        for i in self.fake_SB_ves:
            self.loss_ves += self.ce_loss(i, self.real_SB_ves_label)
        self.loss_ves_cons = 0
        for i, j in zip(self.fake_SB_ves, self.fake_TB_ema_ves):
            self.loss_ves_cons += self.consistency_criterion(torch.log(self.softmax_2d(i)), torch.log(self.softmax_2d(j)))

        # self.loss_char = 0
        # self.loss_edge = 0
        # self.loss_ves = 0
        # self.loss_ema = 0
        # self.loss_ves_cons = 0

        self.loss_supervised = self.loss_char + self.loss_edge + self.loss_ves

        consistency_weight = self.get_current_consistency_weight(self.current_epoch)
        self.loss_consistency = consistency_weight * (self.loss_ema + self.loss_ves_cons)

        # TODO: out的rate
        self.loss_G = (self.loss_supervised + self.loss_consistency) * self.opt.lambda_G
        self.loss_G.backward()

    def compute_visuals(self):
        self.fake_SB_stage2, self.fake_SB_stage1 = self.fake_SB[0], self.fake_SB[1]
        self.fake_TB_stage2, self.fake_TB_stage1 = self.fake_TB[0], self.fake_TB[1]
        self.fake_TB_ema_stage2, self.fake_TB_ema_stage1 = self.fake_TB_ema[0], self.fake_TB_ema[1]

        self.fake_TB_ema_ves_vis = self.fake_TB_ema_ves[0].argmax(dim=1).unsqueeze(1)
        self.fake_SB_ves_vis = self.fake_SB_ves[0].argmax(dim=1).unsqueeze(1)
        self.fake_TB_ves_vis = self.fake_TB_ves[0].argmax(dim=1).unsqueeze(1)

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        update_ema_variables(self.netG, self.netG_ema, self.EMA_DECAY, self.current_epoch)

