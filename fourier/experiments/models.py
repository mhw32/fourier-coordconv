from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    get_conv_output_dim,
    bernoulli_log_pdf,
    logistic_256_log_pdf,
    gaussian_log_pdf,
    unit_gaussian_log_pdf,
    log_mean_exp
)
from . import CONV_OPTIONS, DIST_OPTIONS, LABEL_OPTIONS


# --- FourierCoordConv implementation --- 


class ApplyFourierCoordinates(object):
    r"""Appply Fourier encodings representing each position. Does
    this have theoretical benefits over raw positional encodings?

    Inspiration from AIAYN - https://arxiv.org/pdf/1706.03762

    Args:
        d (integer): Number of encoding dimensions. For example
            this could be image height or width.
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2), H_{in}, W_{in})`
    """
    def incorportate_fourier_coords(self, image, xx_channel, yy_channel):
        raise Exception('Must be implemented in subclass')

    def __call__(self, image):
        """
        Args:
            image: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, c_dim, x_dim, y_dim = image.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        # need number of channels
        xx_channel = xx_channel.repeat(c_dim, 1, 1)
        yy_channel = yy_channel.repeat(c_dim, 1, 1)

        xx_channel = xx_channel.float()
        yy_channel = yy_channel.float()

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        # add fourier transformation
        xx_channel, yy_channel = fourier_encoding(xx_channel, yy_channel)

        # cast to CUDA
        xx_channel = xx_channel.to(image.device)
        yy_channel = yy_channel.to(image.device)

        # add positional encodings to data
        return self.incorportate_fourier_coords(image, xx_channel, yy_channel)


class AddFourierCoordinates(ApplyFourierCoordinates):
    def incorportate_fourier_coords(self, image, xx_channel, yy_channel):
        """ Addition """
        return image + xx_channel + yy_channel


class AddFourierCoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AddFourierCoordConv2d, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)

        self.coord_applier = AddFourierCoordinates()

    def forward(self, x):
        x = self.coord_applier(x)
        x = self.conv_layer(x)

        return x


class AddFourierCoordConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1):
        super(AddFourierCoordConvTranspose2d, self).__init__()

        self.conv_tr_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size, stride=stride,
                                                padding=padding,
                                                output_padding=output_padding,
                                                groups=groups, bias=bias,
                                                dilation=dilation)

        self.coord_applier = AddFourierCoordinates()

    def forward(self, x):
        x = self.coord_applier(x)
        x = self.conv_tr_layer(x)

        return x

class ConcatFourierCoordinates(ApplyFourierCoordinates):
    def incorportate_fourier_coords(self, image, xx_channel, yy_channel):
        """ Concatenation """
        return torch.cat([image, xx_channel, yy_channel], dim=1)


class ConcatFourierCoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConcatFourierCoordConv2d, self).__init__()
        in_channels *= 3
        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)

        self.coord_applier = ConcatFourierCoordinates()

    def forward(self, x):
        x = self.coord_applier(x)
        x = self.conv_layer(x)

        return x


class ConcatFourierCoordConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1):
        super(ConcatFourierCoordConvTranspose2d, self).__init__()
        in_channels *= 3
        self.conv_tr_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size, stride=stride,
                                                padding=padding,
                                                output_padding=output_padding,
                                                groups=groups, bias=bias,
                                                dilation=dilation)

        self.coord_applier = ConcatFourierCoordinates()

    def forward(self, x):
        x = self.coord_applier(x)
        x = self.conv_tr_layer(x)

        return x

def fourier_encoding(xx_positions, yy_positions):
    r"""Given a matrix of positions, convert to sine/cosine 
    frequencies using odd/even positions.

        PE(pos, 2i)   = sin(pos/10000^(2i/d))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d))
    """
    # let d be the number of channels
    _, d_hid, image_size, _ = xx_positions.size()
    xx_positions_npy = xx_positions.numpy()
    yy_positions_npy = yy_positions.numpy()

    def get_sinusoid_encoding_table(n_position, d_hid):
	''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return sinusoid_table

    i_vec = get_sinusoid_encoding_table(image_size, d_hid)
    i_mask = np.stack([i_vec for _ in xrange(image_size)])
    i_mask = np.rollaxis(i_mask, 2)
    j_mask = np.transpose(i_mask, (0, 2, 1))

    xx_positions = torch.from_numpy(i_mask).float()
    yy_positions = torch.from_numpy(j_mask).float()

    return xx_positions, yy_positions


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
        """
        Args:
            image: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = image.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        ret = torch.cat([
            image,
            xx_channel.type_as(image),
            yy_channel.type_as(image)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(image) - 0.5, 2) + 
                            torch.pow(yy_channel.type_as(image) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


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
    'AddFourier': AddFourierCoordConv2d,
    'ConcatFourier': ConcatFourierCoordConv2d,
}

CONV_TRANS_FUNCS = {
    'vanilla': nn.ConvTranspose2d,
    'coord': CoordConvTranspose2d,
    'AddFourier': AddFourierCoordConvTranspose2d,
    'ConcatFourier': ConcatFourierCoordConvTranspose2d,
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
    @param conv: string [default: vanilla]
                 vanilla|coord
    """
    def __init__(self, n_channels, image_size, z_dim, n_filters=64, conv='vanilla'):
        super(Encoder, self).__init__()
        assert conv in CONV_OPTIONS, "conv %s not supported." % conv
        assert image_size in [28, 32], "reshape image to be either 28x28 or 32x32"

        self.z_dim = z_dim
        self.n_channels = n_channels
        self.image_size = image_size
        self.n_filters = n_filters
        
        if self.image_size == 28:
            self.conv_layers = gen_28_conv_layers(
                CONV_FUNCS[conv], self.n_channels, self.n_filters)
            self.cout = gen_28_conv_output_dim(self.image_size)
        elif self.image_size == 32:
            self.conv_layers = gen_32_conv_layers(
                CONV_FUNCS[conv], self.n_channels, self.n_filters)
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
                 conv='vanilla', dist='bernoulli'):
        super(Decoder, self).__init__()
        assert conv in CONV_OPTIONS, "conv %s not supported." % conv
        assert image_size in [28, 32], "reshape image to be either 28x28 or 32x32"
        assert dist in DIST_OPTIONS, "dist %s not supported." % dist

        self.z_dim = z_dim
        self.dist = dist
        self.n_channels = n_channels
        self.image_size = image_size
        self.n_filters = n_filters

        if self.image_size == 28:
            self.conv_layers = gen_28_deconv_layers(CONV_FUNCS[conv], CONV_TRANS_FUNCS[conv], 
                                                    self.n_channels, self.n_filters, dist=self.dist)
            self.cout = gen_28_conv_output_dim(self.image_size)
        elif self.image_size == 32:
            self.conv_layers = gen_32_deconv_layers(CONV_FUNCS[conv], CONV_TRANS_FUNCS[conv], 
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
            x_mu = torch.sigmoid(h)
            return x_mu
        elif self.dist == 'gaussian':
            x_mu = torch.sigmoid(h[:, 0].unsqueeze(1))
            x_logvar = F.hardtanh(h[:, 1].unsqueeze(1), min_val=-4.5,max_val=0.)
            return x_mu, x_logvar


# --- main model declarations ---


class VAE(nn.Module):
    def __init__(self, n_channels, image_size, z_dim, n_filters=64, 
                    conv='vanilla', dist='bernoulli'):
        super(VAE, self).__init__()
        assert conv in CONV_OPTIONS, "conv %s not supported." % conv
        assert dist in DIST_OPTIONS, "dist %s not supported." % dist
        
        self.n_channels = n_channels
        self.image_size = image_size
        self.z_dim = z_dim
        self.dist = dist
        self.conv = conv
        self.n_filters = n_filters
        self.encoder = Encoder(self.n_channels, self.image_size, self.z_dim,
                                n_filters=self.n_filters, conv=self.conv)
        self.decoder = Decoder(self.n_channels, self.image_size, self.z_dim,
                                n_filters=self.n_filters, conv=self.conv, 
                                dist=self.dist)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return eps.mul(std).add_(mu)

    def forward(self, data):
        z_mu, z_logvar = self.encoder(data)
        z = self.reparameterize(z_mu, z_logvar)

        if self.dist == 'bernoulli':
            recon_data_mu = self.decoder(z)
            return data, recon_data_mu, z, z_mu, z_logvar
        elif self.dist == 'gaussian':
            recon_data_mu, recon_data_logvar = self.decoder(z)
            return data, recon_data_mu, recon_data_logvar, z, z_mu, z_logvar
        else:
            raise Exception('dist %s not recognized.' % self.dist)

    def get_marginal(self, data, n_samples=100):
        batch_size =  data.size(0)
        z_mu, z_logvar = self.encoder(data)
        data_shape = (data.size(1), data.size(2), data.size(3))
        data_2d = data.view(batch_size, np.prod(data_shape))

        log_w = []
        for i in xrange(n_samples):
            z = self.reparameterize(z_mu, z_logvar)

            if self.dist == 'bernoulli':
                recon_x_mu = self.decoder(z)
                recon_x_mu_2d = recon_x_mu.view(batch_size, np.prod(data_shape))
                log_p_x_given_z = bernoulli_log_pdf(data_2d, recon_x_mu_2d)
            elif self.dist == 'gaussian':
                recon_x_mu, recon_x_logvar = self.decoder(z)
                recon_x_mu_2d = recon_x_mu.view(batch_size, np.prod(data_shape))
                recon_x_logvar_2d = recon_x_logvar.view(batch_size, np.prod(data_shape))
                log_p_x_given_z = logistic_256_log_pdf(data_2d, recon_x_mu_2d, recon_x_logvar_2d)

            log_q_z_given_x = gaussian_log_pdf(z, z_mu, z_logvar)
            log_p_z = unit_gaussian_log_pdf(z)

            log_w_i = log_p_x_given_z + log_p_z - log_q_z_given_x
            log_w.append(log_w_i.unsqueeze(1))

        log_w = torch.cat(log_w, dim=1)
        log_p_x = log_mean_exp(log_w, dim=1)
        log_p = -torch.mean(log_p_x)

        return log_p


class Classifier(nn.Module):
    def __init__(self, n_channels, image_size, n_class, hidden_dim=256, n_filters=64, 
                    conv='vanilla', label_dist='bernoulli'):
        super(Classifier, self).__init__()
        assert conv in CONV_OPTIONS, "conv %s not supported." % conv
        assert label_dist in LABEL_OPTIONS, "label_dist %s not supported." % label_dist

        self.n_channels = n_channels
        self.image_size = image_size
        self.n_class = n_class
        self.conv = conv
        self.n_filters = n_filters
        self.label_dist = label_dist
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(self.n_channels, self.image_size, self.hidden_dim,
                                n_filters=self.n_filters, conv=self.conv)
        self.classifier = nn.Linear(self.hidden_dim, n_class)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return eps.mul(std).add_(mu)

    def forward(self, data):
        h, _ = self.encoder(data)
        h = F.relu(h)
        output = self.classifier(h)
        
        if self.label_dist == 'bernoulli':
            output = torch.sigmoid(output)
        elif self.label_dist == 'categorical':
            output = F.log_softmax(output, dim=1)
        else:
            raise Exception('label_dist %s not supported.' % self.label_dist)

        return output
