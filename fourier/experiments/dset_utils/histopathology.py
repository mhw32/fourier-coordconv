from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle
import numpy as np

import torch
import torch.utils.data as data

from .. import DATA_DIR


class Histopathology(data.Dataset):
    r"""Grayscale Histopathology Dataset.

    See https://arxiv.org/pdf/1611.09630.pdf

    @param split: string
                  training|validation|test
    """
    def __init__(self, data_dir, split='training'):
        super(Histopathology, self).__init__()
        self.pickle_path = os.path.join(data_dir, 'histopathology/histopathology.pkl')

        with open(self.pickle_path) as fp:
            self.data = cPickle.load(fp)

        assert split in self.data.keys(), "<split> not recognized."
        self.data = self.data[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = np.clip(image, 1./512, 1.-1./512)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        return image, index
