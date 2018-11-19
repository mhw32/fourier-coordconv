from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import math
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from mime.experiments import DIST_OPTIONS

LOG2PI = float(np.log(2.0 * math.pi))


# --- wrapper over ELBO functions for training ---


def elbo(out, dist='bernoulli', annealing_factor=1.):
    r"""Wrapper around gaussian_image_elbo and bernoulli_image_elbo."""
    assert dist in DIST_OPTIONS
    if dist == 'bernoulli':
        (x, recon_x_mu, z, z_mu, z_logvar) = out
        elbo = bernoulli_elbo(x, recon_x_mu, z, z_mu, z_logvar,
                              annealing_factor=annealing_factor)
    elif dist == 'gaussian':
        (x, recon_x_mu, recon_x_logvar, z, z_mu, z_logvar) = out
        elbo = gaussian_elbo(x, recon_x_mu, recon_x_logvar, z, z_mu, z_logvar,
                                annealing_factor=1.)
    else:
        raise Exception('dist %s not recognized.' % dist)

    return elbo


# --- evidence lower bound definitions ---


def bernoulli_elbo(x, recon_x_mu, z, z_mu, z_logvar, annealing_factor=1.):
    r"""Lower bound on x evidence (bernoulli parameterization).

    @param x: torch.Tensor
                observation of an x
    @param recon_x_mu: torch.Tensor
                        tensor of logits representing each pixel.
    @param z: torch.Tensor
              latent sample
    @param z_mu: torch.Tensor
                 mean of variational distribution
    @param z_logvar: torch.Tensor
                     log-variance of variational distribution
    """
    n, c, h, w = x.size()
    x_2d = x.view(n, c * h * w)
    recon_x_mu_2d = recon_x_mu.view(n, c * h * w)
    BCE = -bernoulli_log_pdf(x_2d, recon_x_mu_2d)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    KLD = torch.sum(KLD, dim=1)

    # lower bound on marginal likelihood
    ELBO = BCE + annealing_factor * KLD
    ELBO = torch.mean(ELBO)

    return ELBO


def gaussian_elbo(x, recon_x_mu, recon_x_logvar, z, z_mu, z_logvar,
                    annealing_factor=1.):
    r"""Lower bound on x evidence (logistic256 parameterization).

    @param x: torch.Tensor
                  observation of an x
    @param recon_x_mu: torch.Tensor
                           tensor of means for each pixel.
    @param recon_x_logvar: torch.Tensor
                               tensor of log variances for each pixel.
    @param z: torch.Tensor
              latent sample
    @param mu: torch.Tensor
               mean of variational distribution
    @param logvar: torch.Tensor
                   log-variance of variational distribution
    """
    n, c, h, w = x.size()
    x_2d = x.view(n, c * h * w)
    recon_x_mu_2d = recon_x_mu.view(n, c * h * w)
    recon_x_logvar_2d = recon_x_logvar.view(n, c * h * w)
    RECON = -logistic_256_log_pdf(x_2d, recon_x_mu_2d, recon_x_logvar_2d)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    KLD = torch.sum(KLD, dim=1)

    # lower bound on marginal likelihood
    ELBO = RECON + annealing_factor * KLD
    ELBO = torch.mean(ELBO)

    return ELBO


# --- probability distributions ---


def bernoulli_log_pdf(x, mu):
    r"""Log-likelihood of data given ~Bernoulli(mu)

    @param x: PyTorch.Tensor
              ground truth input
    @param mu: PyTorch.Tensor
               Bernoulli distribution parameters
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


def logistic_256_log_pdf(x, mean, logvar):
    r"""In practice it is problematic to use a gaussian decoder b/c it will
    memorize the data (defaulting to a regular decoder). Constraining it as
    a discrete space is important.
    https://www.reddit.com/r/MachineLearning/comments/4eqifs/gaussian_observation_vae/
    """
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = torch.log(cdf_plus - cdf_minus + 1.e-7)

    log_pdf = torch.sum(log_logist_256, 1)

    return log_pdf


def gaussian_log_pdf(x, mu, logvar):
    r"""Log-likelihood of data given ~N(mu, exp(logvar))
    log f(x) = log(1/sqrt(2*pi*var) * e^(-(x - mu)^2 / var))
             = -1/2 log(2*pi*var) - 1/2 * ((x-mu)/sigma)^2
             = -1/2 log(2pi) - 1/2log(var) - 1/2((x-mu)/sigma)^2
             = -1/2 log(2pi) - 1/2[((x-mu)/sigma)^2 + log var]
    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -LOG2PI * x.size(1) / 2. - \
        torch.sum(logvar + torch.pow(x - mu, 2) / (torch.exp(logvar) + 1e-7), dim=1) / 2.

    return log_pdf


def unit_gaussian_log_pdf(x):
    r"""Log-likelihood of data given ~N(0, 1)
    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -LOG2PI * x.size(1) / 2. - \
        torch.sum(torch.pow(x, 2), dim=1) / 2.

    return log_pdf


# --- other helpful utilities ---


def make_one_hot(x, n_class):
    x = x.long()
    x_1hot = torch.FloatTensor(x.size(0), n_class)
    if x.is_cuda:
        x_1hot = x_1hot.cuda()
    x_1hot.zero_()
    x_1hot.scatter_(1, x.unsqueeze(1), 1)

    return x_1hot


def log_mean_exp(x, dim=1):
    r"""log(1/k * sum(exp(x))): this normalizes x.

    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


def get_image_coords(size):
    maskH = torch.arange(0, size).unsqueeze(1).repeat(1, size)
    maskW = torch.arange(0, size).unsqueeze(1).repeat(1, size).t()
    return maskH, maskW


def get_conv_output_dim(I, K, P, S):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2*P)/float(S) + 1
    return int(O)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))
