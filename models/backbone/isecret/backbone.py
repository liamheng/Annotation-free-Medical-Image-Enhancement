# -*- coding: utf-8 -*-
from torch.nn.modules.pooling import AdaptiveAvgPool2d
import torch.nn as nn
# from models.backbone.isecret.model.utils import make_norm, make_paddding, check_architecture
# from models.backbone.isecret.model.blocks import ResnetBlock
import torch
import torch.nn.functional as F
import numpy as np
import os
from models.networks import init_net, get_norm_layer
from packaging import version

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding, norm_layer, use_bias, use_dropout=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding (nn.Padding)  -- the instance of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        conv_block += [padding(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [padding(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ImportanceResGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_downs=2, n_filters=64, n_blocks=9):
        nn.Module.__init__(self)

        # Get padding
        padding = nn.ReflectionPad2d

        # Get norm
        norm_layer, use_bias = nn.InstanceNorm2d, True

        # Build Head
        head = [padding(3),
                nn.Conv2d(input_nc, n_filters, kernel_size=7, bias=use_bias),
                norm_layer(n_filters),
                nn.ReLU(True)]

        # Build down-sampling
        downs = []
        for i in range(n_downs):
            mult = 2 ** i
            downs += [padding(1), nn.Conv2d(n_filters * mult,
                                            n_filters * mult * 2,
                                            kernel_size=3, stride=2, bias=use_bias),
                      norm_layer(n_filters * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downs

        neck = []
        # Build res-blocks
        self.in_ch = n_filters * mult * 4
        for i in range(n_blocks):
            neck += [ResnetBlock(n_filters * mult, padding=padding,
                                norm_layer=norm_layer, use_dropout=False,
                                use_bias=use_bias)]

        ups = []
        # Build up-sampling
        for i in range(n_downs):
            mult = 2 ** (n_downs - i)
            ups += [nn.ConvTranspose2d(n_filters * mult,
                                      int(n_filters * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(n_filters * mult / 2)),
                   nn.ReLU(True)]



        importance_ups = []
        # Build unctainty-aware up-sampling
        for i in range(n_downs):
            mult = 2 ** (n_downs - i)
            importance_ups += [nn.ConvTranspose2d(n_filters * mult,
                                      int(n_filters * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(n_filters * mult / 2)),
                   nn.ReLU(True)]

        
        # Build tail
        ups += [padding(3)]
        ups += [nn.Conv2d(n_filters, output_nc, kernel_size=7, padding=0)]

        ups += [nn.Tanh()]

        # Build importance tail
        importance_ups += [padding(3)]
        importance_ups += [nn.Conv2d(n_filters, output_nc, kernel_size=7, padding=0)]

        # Make model
        self.head = nn.Sequential(*head)
        self.downs = nn.Sequential(*downs)
        self.neck = nn.Sequential(*neck)
        self.ups = nn.Sequential(*ups)
        self.importance_ups = nn.Sequential(*importance_ups)

    def forward(self, input, need_importance=False, layers=None):
        if layers is None:
            x = self.head(input)
            x = self.downs(x)
            x = self.neck(x)
            output = self.ups(x)
            if need_importance:
                importance = self.importance_ups(x)
                return output, importance
            else:
                return output
        else:
            return self.forward_features(input, layers)

    def forward_features(self, input, layers):
        # We only focus on the encoding part
        feat = input
        feats = []
        layer_id = 0
        for layer in self.head:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        for layer in self.downs:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        for layer in self.neck:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        return feats, feat


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class Standardlize(nn.Module):
    def __init__(self):
        super(Standardlize, self).__init__()

    def forward(self, x):
        return x / x.mean()

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp, gpu_ids, init_type='xavier', init_gain=0.02, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        nn.Module.__init__(self)
        self.l2norm = Normalize(2)
        self.standard = Standardlize() # Stable gradient
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, weight_map=None):
        return_ids = []
        return_feats = []
        return_weight_samples = []
        weight_sample = None
        if self.use_mlp and not self.mlp_init:
            print('[INFO] Create MLP...')
            self.create_mlp(feats)
            self.mlp_init = True
            return
        if weight_map is not None:
            weight_map = weight_map.mean(dim=[1], keepdim=False).unsqueeze(dim=1)
            weight_map = self.standard(torch.exp(-weight_map))
        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape[0], feat.shape[1],  feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if weight_map is not None:
                weight_map = F.interpolate(weight_map, size=(W, H), mode='area')
                weight_map_reshape = weight_map.permute(0, 2, 3, 1).flatten(1, 2)
                weight_sample = weight_map_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])

            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
                if weight_map is not None:
                    weight_sample = weight_sample.permute(0, 2, 1).reshape([B, weight_sample.shape[-1], H, W])
            return_feats.append(x_sample)
            if weight_map is not None:
                return_weight_samples.append(weight_sample)
        if weight_map is not None:
            return return_feats, return_ids, return_weight_samples
        return return_feats, return_ids

class ISLoss(nn.Module):
    # Importance supervised loss (IS-loss)
    def __init__(self,  reduction='mean'):
        super(ISLoss, self).__init__()
        self.reduction = reduction

    def forward(self, source, target, weight):
        loss = self.calculate_loss(source, target, weight)
        return loss

    def calculate_loss(self, pred, target, log_weight):
        weight = torch.exp(-log_weight)
        mse = F.mse_loss(pred, target)
        if self.reduction == 'mean':
            return torch.mean(weight * mse + log_weight)
        return weight * mse + log_weight

class LSGANLoss(nn.Module):
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.real_loss = lambda x: F.mse_loss(x, torch.ones_like(x))
        self.fake_loss = lambda x: F.mse_loss(x, torch.zeros_like(x))

    def update_g(self, good_dis, fake_good, bad_dis=None, fake_bad=None):
        fake_good_logits = good_dis(fake_good)
        good_loss = self.real_loss(fake_good_logits)
        if bad_dis is None:
            return good_loss
        fake_bad_logits = bad_dis(fake_bad)
        bad_loss = self.real_loss(fake_bad_logits)
        return good_loss, bad_loss

    def update_d(self, good_dis, real_good, fake_good, bad_dis=None, real_bad=None, fake_bad=None):
        # Train dis_good
        real_good_logits = good_dis(real_good)
        fake_good_logits = good_dis(fake_good.detach())
        real_good_loss = self.real_loss(real_good_logits)
        fake_good_loss = self.fake_loss(fake_good_logits)
        good_dis_loss = (real_good_loss + fake_good_loss) / 2
        return good_dis_loss

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, weight=1.0):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        # Calculate the bmm of the f_q and f_k

        # N x patches X 1 X C
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size
        # neg logit -- current batch
        # reshape features to batch size
        # B x patches x C
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        # feat_q = feat_q.view(self.opt.batch_size_per_gpu, -1, dim)
        # feat_k = feat_k.view(self.args.dist.batch_size_per_gpu, -1, dim)

        # h x w
        npatches = feat_q.size(1)

        # B x patches x C X B x C x patches -> B x patches x patches
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)  # Replace the corresponding patch with negative value

        # B x patches X patches
        l_neg = l_neg_curbatch.view(-1, npatches)
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device)) * weight

        return loss