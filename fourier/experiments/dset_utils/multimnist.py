r"""This script generates a dataset similar to the Multi-MNIST dataset
described in [1].

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import random
import numpy as np
from PIL import Image
from random import shuffle
from scipy.misc import imresize
from tqdm import tqdm

import torch
import torch.utils.data as data

from .. import DATA_DIR
from torchvision.datasets import MNIST
from torchvision import transforms

sys.setrecursionlimit(100000)


def load_dynamic_mnist_test_set(data_dir):
    # initial load we can take advantage of the dataloader
    test_loader = data.DataLoader(
        MNIST(os.path.join(data_dir, 'mnist'), train=False, transform=transforms.ToTensor()),
        batch_size=100, shuffle=True)

    # load it back into numpy tensors...
    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    y_test = np.array(test_loader.dataset.test_labels.float().numpy(), dtype=int)

    # binarize once!!! (we don't dynamically binarize this)
    np.random.seed(777)
    x_test = np.random.binomial(1, x_test)

    x_test = torch.from_numpy(x_test).float().unsqueeze(1)
    y_test = torch.from_numpy(y_test)

    # pytorch data loader
    test_dataset = data.TensorDataset(x_test, y_test)

    return test_dataset


def load_dynamic_multimnist_test_set(data_dir, fix_digit_positions=False):
    # initial load we can take advantage of the dataloader
    test_loader = data.DataLoader(
        MultiMNIST(data_dir, train=False, fix_digit_positions=fix_digit_positions,
                   transform=transforms.ToTensor()),
        batch_size=100, shuffle=True)

    # load it back into numpy tensors...
    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    y_test = np.array(test_loader.dataset.test_labels, dtype=int)

    # binarize once!!! (we don't dynamically binarize this)
    np.random.seed(777)
    x_test = np.random.binomial(1, x_test)

    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test)

    # pytorch data loader
    test_dataset = data.TensorDataset(x_test, y_test)

    return test_dataset


class MultiMNIST(data.Dataset):
    r"""Images with 0 to N digits of (hopefully) non-overlapping MNIST numbers.

    @param fix_digit_positions: boolean [default: False]
                                use fixed location of images.
    """
    processed_folder = 'multimnist'
    fixed_data_folder = 'multimnist_fixed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 fix_digit_positions=False):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.data_folder = (self.fixed_data_folder if fix_digit_positions else
                            self.processed_folder)

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.data_folder, self.training_file))
            self.train_data = self.train_data.float() / 255.
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.data_folder, self.test_file))
            self.test_data = self.test_data.float() / 255.

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.data_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.data_folder, self.test_file))


def sample_one(canvas_size, mnist, resize=True, translate=True):
    i = np.random.randint(mnist['digits'].shape[0])
    digit = mnist['digits'][i]
    label = mnist['labels'][i]
    if resize:  # resize only if user specified
        scale = 0.1 * np.random.randn() + 2.0
        resized = imresize(digit, 1. / scale)
    else:
        resized = digit
    w = resized.shape[0]
    assert w == resized.shape[1]
    padding = canvas_size - w
    if translate:  # translate only if user specified
        pad_l = np.random.randint(0, padding)
        pad_r = np.random.randint(0, padding)
        pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
        positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    else:
        pad_l = padding // 2
        pad_r = padding // 2
        pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
        positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    return positioned, label


def sample_multi(num_digits, canvas_size, mnist, resize=True, translate=True):
    canvas = np.zeros((canvas_size, canvas_size))
    labels = []
    for _ in range(num_digits):
        positioned_digit, label = sample_one(canvas_size, mnist, resize=resize,
                                             translate=translate)
        canvas += positioned_digit
        labels.append(label)

    # Crude check for overlapping digits.
    if np.max(canvas) > 255:
        return sample_multi(num_digits, canvas_size, mnist,
                            resize=resize, translate=translate)
    else:
        return canvas, labels


def mk_dataset(n, mnist, min_digits, max_digits, canvas_size,
               resize=True, translate=True):
    x = []
    y = []
    for _ in tqdm(range(n)):
        num_digits = np.random.randint(min_digits, max_digits + 1)
        canvas, labels = sample_multi(num_digits, canvas_size, mnist,
                                      resize=resize, translate=translate)
        x.append(canvas)
        y.append(labels)
    return np.array(x, dtype=np.uint8), y


def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(root=os.path.join(DATA_DIR, 'mnist'), train=True, download=True))

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(root=os.path.join(DATA_DIR, 'mnist'), train=False, download=True))

    train_data = {
        'digits': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels
    }

    test_data = {
        'digits': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels
    }

    return train_data, test_data


def make_dataset(root, folder, training_file, test_file, min_digits=0, max_digits=2,
                 resize=True, translate=True):
    if not os.path.isdir(os.path.join(root, folder)):
        os.makedirs(os.path.join(root, folder))

    np.random.seed(681307)
    train_mnist, test_mnist = load_mnist()
    train_x, train_y = mk_dataset(60000, train_mnist, min_digits, max_digits, 32,
                                  resize=resize, translate=translate)
    test_x, test_y = mk_dataset(10000, test_mnist, min_digits, max_digits, 32,
                                resize=resize, translate=translate)

    train_x = torch.from_numpy(train_x).byte().unsqueeze(1)
    test_x = torch.from_numpy(test_x).byte().unsqueeze(1)

    training_set = (train_x, train_y)
    test_set = (test_x, test_y)

    with open(os.path.join(root, folder, training_file), 'wb') as f:
        torch.save(training_set, f)

    with open(os.path.join(root, folder, test_file), 'wb') as f:
        torch.save(test_set, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', action='store_true', default=False,
                        help='If True, ignore resize/translate options and generate')
    args = parser.parse_args()

    # Generate the training set and dump it to disk. (Note, this will
    # always generate the same data, else error out.)
    make_dataset(DATA_DIR, 'multimnist', 'training.pt', 'test.pt',
                    min_digits=1, max_digits=1, resize=True, translate=True)
