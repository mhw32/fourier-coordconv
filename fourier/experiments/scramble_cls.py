r"""Scramble the coordinates and compare performance."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
from tqdm import tqdm

import torch
from .models import Classifier
from .datasets import build_dataset
from . import (SUP_DATA_OPTIONS, DATA_SHAPE, DATA_DIR, DATA_DIST, 
                DATA_LABEL_NUM, DATA_LABEL_DIST)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str,
                        help='where to load a model')
    parser.add_argument('--scramble', action='store_true', default=False,
                        help='scramble the coordinate layers [default: False]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['state_dict']
    train_args = checkpoint['args']

    # for reproducibility
    torch.manual_seed(train_args.seed)
    np.random.seed(train_args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    test_dataset = build_dataset(train_args.dataset, DATA_DIR, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_args.batch_size, shuffle=False)

    n_channels, image_size, image_size = DATA_SHAPE[train_args.dataset]
    model = Classifier(n_channels, image_size, DATA_LABEL_NUM[train_args.dataset], 
                        n_filters=train_args.n_filters, hidden_dim=train_args.hidden_dim, 
                        conv=train_args.conv, label_dist=DATA_LABEL_DIST[train_args.dataset])
    model = model.load_state_dict(state_dict)
    if args.scramble:
        model.conv_layers[0].scramble = True
    model = model.to(device)

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for data, label in test_loader:
            batch_size = len(data)
            data = data.to(device)
            label = label.to(device)

            out = model(data)
            loss = F.nll_loss(out, label)

            pred = torch.exp(out).max(1)[1]
            accuracy = accuracy_score(label.cpu().numpy(), pred.cpu().numpy())

            accuracy_meter.update(accuracy, batch_size)
            pbar.update()
        pbar.close()

    print('====> Test Epoch: {}\tAccuracy: {:.4f}'.format(
        epoch, accuracy_meter.avg))
