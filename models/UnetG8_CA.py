import torch
import torch.nn as nn
import functools
from modules.layers import CBAM
from models.guided_filter_pytorch.HFC_filter import HFCFilter

# from torchvision.models import resnet18

class UnetG8_CA(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, reduction_ratio=16):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                images of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetG8_CA, self).__init__()
        self.hfc_filter = HFCFilter(21, 20)
        self.hfc_pool = nn.AvgPool2d(2)
        unet_block8 = UnetCASkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                                norm_layer=norm_layer, innermost=True)  # add the innermost layer
        unet_block7 = UnetCASkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block6 = UnetCASkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block5 = UnetCASkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block4 = UnetCASkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)
        unet_block3 = UnetCASkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        unet_block2 = UnetCASkipConnectionBlock(ngf, ngf * 2, input_nc=ngf+3, norm_layer=norm_layer)
        unet_block1 = UnetCASkipConnectionBlock(output_nc, ngf, input_nc=6, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        # TODO:cbam模块是在后头，修改位置
        self.down1, self.up1, self.cbam1 = unet_block1.down, unet_block1.up, unet_block1.cbam
        self.down2, self.up2, self.cbam2 = unet_block2.down, unet_block2.up, unet_block2.cbam
        self.down3, self.up3, self.cbam3 = unet_block3.down, unet_block3.up, unet_block3.cbam
        self.down4, self.up4, self.cbam4 = unet_block4.down, unet_block4.up, unet_block4.cbam
        self.down5, self.up5, self.cbam5 = unet_block5.down, unet_block5.up, unet_block5.cbam
        self.down6, self.up6, self.cbam6 = unet_block6.down, unet_block6.up, unet_block6.cbam
        self.down7, self.up7, self.cbam7 = unet_block7.down, unet_block7.up, unet_block7.cbam
        self.down8, self.up8, self.cbam8 = unet_block8.down, unet_block8.up, unet_block8.cbam

    def forward(self, x, mask):
        """Standard forward"""
        # downsample
        hfc0 = self.hfc_filter(x, mask)
        d1 = self.down1(torch.cat([x, hfc0], 1))
        c1 = self.cbam1(d1)
        hfc1 = self.hfc_pool(hfc0)
        d2 = self.down2(torch.cat([d1, hfc1], 1))
        c2 = self.cbam2(d2)
        d3 = self.down3(d2)
        c3 = self.cbam3(d3)
        d4 = self.down4(d3)
        c4 = self.cbam4(d4)
        d5 = self.down5(d4)
        c5 = self.cbam5(d5)
        d6 = self.down6(d5)
        c6 = self.cbam6(d6)
        d7 = self.down7(d6)
        c7 = self.cbam7(d7)

        d8 = self.down8(d7)
        c8 = self.cbam8(d8)

        # upsample
        u7 = self.up8(c8)
        # u7 = self.up8(d8)

        u6 = self.up7(torch.cat([u7, c7], 1))
        u5 = self.up6(torch.cat([u6, c6], 1))
        u4 = self.up5(torch.cat([u5, c5], 1))
        u3 = self.up4(torch.cat([u4, c4], 1))
        u2 = self.up3(torch.cat([u3, c3], 1))
        u1 = self.up2(torch.cat([u2, c2], 1))
        out = self.up1(torch.cat([u1, c1], 1))
        out_hfc = self.hfc_filter(out, mask)

        # return u1
        return out, hfc0, out_hfc


class UnetCASkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, reduction_ratio=16):
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
        super(UnetCASkipConnectionBlock, self).__init__()
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
        # CA应该加载bn之后，ref https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        cbam = CBAM(inner_nc, reduction_ratio=reduction_ratio)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            # downconv_input = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
            #                      stride=1, padding=1, bias=use_bias)
            # downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
            #                      stride=2, padding=1, bias=use_bias)
            # downnorm = norm_layer(inner_nc)
            # upconv = nn.ConvTranspose2d(inner_nc * 2, inner_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            # upconv_optput = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            #
            # down = [downconv_input, downnorm, nn.ReLU(), downconv]
            # up = [uprelu, upconv, nn.ReLU(), upconv_optput, nn.Tanh()]
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
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
        self.cbam = cbam
