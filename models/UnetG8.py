# -*- coding: UTF-8 -*-
"""
@Function:
@File: Unet_G.py
@Date: 2021/7/27 21:31 
@Author: Hever
"""
import torch
import torch.nn as nn
import functools
from models.guided_filter_pytorch.HFC_filter import HFCFilter


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.outc = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         return self.outc(x)


class UnetG8(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetG8, self).__init__()
        self.hfc_filter = HFCFilter(21, 20, sub_mask=True)
        self.hfc_pool = nn.AvgPool2d(2)
        unet_block8 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        # unet_block8 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
        #                                       norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block7 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block6 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block5 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block4 = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)
        unet_block3 = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        # unet_block2 = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        # unet_block1 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        unet_block2 = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=ngf+3, extra_inner_input_nc=None, norm_layer=norm_layer)
        unet_block1 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=6, extra_inner_input_nc=3, outermost=True,
                                              norm_layer=norm_layer)  # add the outermost layer

        self.down1, self.up1 = unet_block1.down, unet_block1.up
        self.down2, self.up2 = unet_block2.down, unet_block2.up
        self.down3, self.up3 = unet_block3.down, unet_block3.up
        self.down4, self.up4 = unet_block4.down, unet_block4.up
        self.down5, self.up5 = unet_block5.down, unet_block5.up
        self.down6, self.up6 = unet_block6.down, unet_block6.up
        self.down7, self.up7 = unet_block7.down, unet_block7.up
        self.down8, self.up8 = unet_block8.down, unet_block8.up
        # self.down9, self.up9 = unet_block9.down, unet_block9.up

        # self.outc = OutConv(ngf, output_nc)


    def forward(self, x, mask):
        """Standard forward"""
        # downsample
        hfc0 = self.hfc_filter(x, mask)
        d1 = self.down1(torch.cat([x, hfc0], 1))
        hfc1 = self.hfc_pool(hfc0)
        d2 = self.down2(torch.cat([d1, hfc1], 1))
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        # d8 = self.down8(d7)

        bm = self.down8(d7)

        # upsample
        # u8 = self.up9(bm)
        u7 = self.up8(bm)
        u6 = self.up7(torch.cat([u7, d7], 1))
        u5 = self.up6(torch.cat([u6, d6], 1))
        u4 = self.up5(torch.cat([u5, d5], 1))
        u3 = self.up4(torch.cat([u4, d4], 1))
        u2 = self.up3(torch.cat([u3, d3], 1))
        # u1 = self.up2(torch.cat([u2, d2], 1))
        u1 = self.up2(torch.cat([u2, d2], 1))
        out = self.up1(torch.cat([u1, d1, hfc1], 1))
        # out = self.up1(torch.cat([u1, d1], 1))  # cat(64+64)-->conv(64)
        # final_out = self.outc(out)  # 最终输出处理
        out_hfc = self.hfc_filter(out, mask)

        return out, hfc0, out_hfc
        # if len(layers) > 0:
        #     return final_out, feats
        # else:
        #     return final_out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, extra_inner_input_nc=None):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)


        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            if extra_inner_input_nc is not None:
                upconv = nn.ConvTranspose2d(inner_nc * 2+extra_inner_input_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            # in_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
            #                     stride=1, padding=1, bias=use_bias)
            # inner_conv = nn.Conv2d(inner_nc*2, inner_nc, kernel_size=3,
            #                      stride=1, padding=1, bias=use_bias)
            down = [downconv]
            # 注意，这样就结构不对称了，应该outer里面加个relu, out_conv
            up = [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, inner_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
        else:
            if extra_inner_input_nc is not None:
                upconv = nn.ConvTranspose2d(inner_nc * 2+extra_inner_input_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]
        self.up = nn.Sequential(*up)
        self.down = nn.Sequential(*down)

if __name__ == '__main__':

    net = UnetG8(input_nc=3,output_nc=3,use_dropout=True)
    net.to('cuda:0')
    input = torch.randn(1, 3, 256, 256).to('cuda:0')
    mask = torch.randn(1, 1, 256, 256).to('cuda:0')

    # mask = torch.randn(1, 3, 256, 256).to('cuda:0')
    output = net(input, mask)
    print(output.shape)