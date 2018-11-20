from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

from torchvision.datasets import \
    MNIST, FashionMNIST, SVHN, CIFAR10
from .dset_utils.histopathology import Histopathology
from .dset_utils.multimnist import (
    load_dynamic_mnist_test_set,
    load_dynamic_multimnist_test_set,
    MultiMNIST,
)
from .dset_utils.celeba import CelebA
from . import DATA_DIR, DATA_OPTIONS


def build_dataset(name, data_dir, train=True):
    r"""Function to build a dataset.

    @param name: string
                 DynamicMNIST|PerturbMNIST|FashionMNIST|Histopathology
                 CelebA|SVHN|CIFAR10
    @param data_dir: string
                     where to find data
    @param train: boolean [default: True]
                  True|False
    """
    assert name in DATA_OPTIONS, \
        "dataset <%s> not recognized." % name
    split = 'train' if train else 'test'

    if name == 'DynamicMNIST':
        if train:
            return MNIST(os.path.join(data_dir, 'mnist'),
                         train=train, download=True,
                         transform=dynamic_binarize)
        else:
            # do not dynamically binarize the test set
            # but dont just take the rounded version (call bernoullli once)
            return load_dynamic_mnist_test_set(data_dir)
    elif name == 'PerturbMNIST':
        if train:
            return MultiMNIST(data_dir, train=train,
                              transform=reshape_and_binarize)
        else:
            return load_dynamic_multimnist_test_set(data_dir)
    elif name == 'FashionMNIST':
        return FashionMNIST(os.path.join(data_dir, 'fashionmnist'),
                            train=train, download=True,
                            transform=transforms.ToTensor())
    elif name == 'Histopathology':
        split = 'training' if train else 'test'
        return Histopathology(data_dir, split=split)
    elif name == 'CIFAR10':
        dataset = CIFAR10(os.path.join(data_dir, 'cifar10'),
                          train=train, download=True,
                          transform=transforms.ToTensor())
    elif name == 'SVHN':
        dataset = SVHN(os.path.join(data_dir, 'svhn'),
                       split=split, download=True,
                       transform=transforms.ToTensor())
    elif name == 'CelebA':
        dataset = CelebA(
            os.path.join(data_dir, 'celeba'), 
            partition=split,
            image_transform=transforms.Compose([transforms.Resize(32),
                                                transforms.CenterCrop(32),
                                                transforms.ToTensor()]),
            attr_transform=None,
        )
    return dataset


def dynamic_binarize(x):
    x = transforms.ToTensor()(x)
    x = torch.bernoulli(x)
    return x


def reshape_and_binarize(x):
    f = transforms.Compose([transforms.Resize(28),
                            transforms.ToTensor()])
    x = f(x)
    x = torch.bernoulli(x)
    return x