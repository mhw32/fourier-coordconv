from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_conv_output_dim
from . import CONV_OPTIONS, DIST_OPTIONS


# --- CoordConv implementation ---
# https://github.com/Wizaron/coord-conv-pytorch


class AddCoordinates(object):
    r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.

    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.

    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`

    Examples:
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)
        >>> device = torch.device("cuda:0")
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_adder(input)
    """

    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        coords = Variable(coords, volatile=image.volatile)
        if image.is_cuda:
            coords = coords.cuda()

        image = torch.cat((coords, image), dim=1)

        return image


class CoordConv2d(nn.Module):
    r"""2D Convolution Module Using Extra Coordinate Information as defined
    in 'An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).

    Args:
        Same as `torch.nn.Conv2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv(input)
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv(input)
        >>> device = torch.device("cuda:0")
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_conv(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=False):
        super(CoordConv2d, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)

        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)

        return x


class CoordConvTranspose2d(nn.Module):
    r"""2D Transposed Convolution Module Using Extra Coordinate Information
    as defined in 'An Intriguing Failing of Convolutional Neural Networks and
    the CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).

    Args:
        Same as `torch.nn.ConvTranspose2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv_tr(input)
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv_tr(input)
        >>> device = torch.device("cuda:0")
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_conv_tr(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, with_r=False):
        super(CoordConvTranspose2d, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_tr_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size, stride=stride,
                                                padding=padding,
                                                output_padding=output_padding,
                                                groups=groups, bias=bias,
                                                dilation=dilation)

        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_tr_layer(x)

        return x


CONV_FUNCS = {
    'vanilla': nn.Conv2d,
    'coord': CoordConv2d,
    # add fourier version here
}

CONV_TRANS_FUNCS = {
    'vanilla': nn.ConvTranspose2d,
    'coord': CoordConvTranspose2d,
    # add fourier version here
}


# --- Encoder and Decoder Architectures ---


def gen_32_conv_layers(conv2d_func, n_channels, n_filters):
    conv_layers = nn.Sequential(
        conv2d_func(n_channels, n_filters, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        conv2d_func(n_filters, n_filters * 2, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters * 2),
        nn.ReLU(),
        conv2d_func(n_filters * 2, n_filters * 4, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters * 4),
        nn.ReLU(),
    )
    return conv_layers


def gen_32_deconv_layers(conv2d_func, conv_trans2d_func, n_channel, n_filters, dist='bernoulli'):
    if dist == 'bernoulli':
        out_channel = n_channel
    elif dist == 'gaussian':
        out_channel = n_channel * 2
    else:
        raise Exception('dist %s not recognized.' % dist)

    conv_layers = nn.Sequential(
        conv_trans2d_func(n_filters * 4, n_filters * 4, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters * 4),
        nn.ReLU(),
        conv_trans2d_func(n_filters * 4, n_filters * 2, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters * 2),
        nn.ReLU(),
        conv_trans2d_func(n_filters * 2, n_filters, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        conv2d_func(n_filters, out_channel, 1, 1, padding=0),
    )
    return conv_layers


def gen_32_conv_output_dim(s):
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    return s


def gen_28_conv_layers(conv2d_func, n_channel, n_filters):
    conv_layers = nn.Sequential(
        conv2d_func(n_channel, n_filters, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        conv2d_func(n_filters, n_filters * 2, 2, 2, padding=1),
        nn.BatchNorm2d(n_filters * 2),
        nn.ReLU(),
        conv2d_func(n_filters * 2, n_filters * 4, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters * 4),
        nn.ReLU(),
    )
    return conv_layers


def gen_28_deconv_layers(conv2d_func, conv_trans2d_func, n_channel, n_filters, dist='bernoulli'):
    if dist == 'bernoulli':
        out_channel = n_channel
    elif dist == 'gaussian':
        out_channel = n_channel * 2
    else:
        raise Exception('dist %s not recognized.' % dist)

    conv_layers = nn.Sequential(
        conv_trans2d_func(n_filters * 4, n_filters * 4, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters * 4),
        nn.ReLU(),
        conv_trans2d_func(n_filters * 4, n_filters * 2, 2, 2, padding=1),
        nn.BatchNorm2d(n_filters * 2),
        nn.ReLU(),
        conv_trans2d_func(n_filters * 2, n_filters, 2, 2, padding=0),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        conv2d_func(n_filters, out_channel, 1, 1, padding=0),
    )
    return conv_layers


def gen_28_conv_output_dim(s):
    s = get_conv_output_dim(s, 3, 1, 2)
    s = get_conv_output_dim(s, 2, 1, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    return s
    

class Encoder(nn.Module):
    r"""Parameterizes q(z|image). Uses DC-GAN architecture.

    https://arxiv.org/abs/1511.06434

    @param n_channels: integer
                       number of input channels.
    @param image_size: integer
                       height and width of input image
    @param z_dim: integer
                  number of latent dimensions.
    @param n_filters: integer [default: 64]
                      number of filters (64 is a LOT)
                      each conv layer progressively blows this up more
    """
    def __init__(self, n_channels, image_size, z_dim, n_filters=64,
                 conv_func='vanilla'):
        super(Encoder, self).__init__()
        assert conv_func in CONV_OPTIONS, "conv_func %s not supported." % conv_func
        assert image_size in [28, 32], "reshape image to be either 28x28 or 32x32"

        self.z_dim = z_dim
        self.n_channels = n_channels
        self.image_size = image_size
        self.n_filters = n_filters
        
        if self.image_size == 28:
            self.conv_layers = gen_28_conv_layers(CONV_FUNCS[conv_func], 
                                                    self.n_channels, self.n_filters)
            self.cout = gen_28_conv_output_dim(self.image_size)
        elif self.image_size == 32:
            self.conv_layers = gen_32_conv_layers(CONV_FUNCS[conv_func],
                                                    self.n_channels, self.n_filters)
            self.cout = gen_32_conv_output_dim(self.image_size)
        else:
            raise Exception('image_size %d not supported.' % self.image_size)
        
        self.fc_layer = nn.Linear(self.n_filters * 4 * self.cout * self.cout, self.z_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)
        h = self.conv_layers(x)
        h = h.view(batch_size, self.n_filters * 4 * self.cout * self.cout)
        h = self.fc_layer(h)
        z_mu, z_logvar = torch.chunk(h, 2, dim=1)

        return z_mu, z_logvar


class Decoder(nn.Module):
    r"""Parameterizes p(image|z). Uses DC-GAN architecture.

    https://arxiv.org/abs/1511.06434
    https://github.com/ShengjiaZhao/InfoVAE/blob/master/model_vae.py

    @param n_channels: integer
                       number of input channels.
    @param image_size: integer
                       height and width of input image
    @param z_dim: integer
                  number of latent dimensions.
    @param n_filters: integer [default: 64]
                      number of filters (64 is a LOT)
                      each conv layer progressively blows this up more
    @param coord_conv: boolean [default: False]
                       add coordinate masks when convolving and deconvolving
    @param dist: string [default: bernoulli]
                 bernoulli|gaussian
    """
    def __init__(self, n_channels, image_size, z_dim, n_filters=64,
                 conv_func='vanilla', dist='bernoulli'):
        super(Decoder, self).__init__()
        assert conv_func in CONV_OPTIONS, "conv_func %s not supported." % conv_func
        assert image_size in [28, 32], "reshape image to be either 28x28 or 32x32"
        assert dist in DIST_OPTIONS, "dist %s not supported." % dist

        self.z_dim = z_dim
        self.dist = dist
        self.n_channels = n_channels
        self.image_size = image_size
        self.n_filters = n_filters

        if self.image_size == 28:
            self.conv_layers = gen_28_deconv_layers(CONV_FUNCS[conv_func], CONV_TRANS_FUNCS[conv_func], 
                                                    self.n_channels, self.n_filters, dist=self.dist)
            self.cout = gen_28_conv_output_dim(self.image_size)
        elif self.image_size == 32:
            self.conv_layers = gen_32_deconv_layers(CONV_FUNCS[conv_func], CONV_TRANS_FUNCS[conv_func], 
                                                    self.n_channels, self.n_filters, dist=self.dist)
            self.cout = gen_32_conv_output_dim(self.image_size)
        else:
            raise Exception('image_size %d not supported.' % self.image_size)

        self.fc_layer = nn.Linear(self.z_dim, self.n_filters * 4 * self.cout * self.cout)

    def forward(self, z):
        batch_size = z.size(0)
        h = F.relu(self.fc_layer(z))
        h = h.view(batch_size, self.n_filters * 4, self.cout, self.cout)
        h = self.conv_layers(h)

        if self.dist == 'bernoulli':
            x_mu = F.sigmoid(h)
            return x_mu
        elif self.dist == 'gaussian':
            x_mu = F.sigmoid(h[:, 0].unsqueeze(1))
            x_logvar = F.hardtanh(h[:, 1].unsqueeze(1), min_val=-4.5,max_val=0.)
            return x_mu, x_logvar
