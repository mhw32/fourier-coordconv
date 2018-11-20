r"""Train a Supervised Classifier."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms

from .utils import AverageMeter, merge_args_with_dict
from .datasets import build_dataset
from .models import Classifier
from .config import CONFIG
from . import (SUP_DATA_OPTIONS, DATA_SHAPE, DATA_DIR, DATA_DIST, 
                DATA_LABEL_NUM, DATA_LABEL_DIST)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='DynamicMNIST|PerturbMNIST|FashionMNIST|CelebA|CIFAR10')
    parser.add_argument('conv', type=str, help='vanilla|coord|AddFourier|ConcatFourier')
    parser.add_argument('out_dir', type=str, help='where to save trained model')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    merge_args_with_dict(args, CONFIG)
    assert args.dataset in SUP_DATA_OPTIONS, "--dataset %s not recognized." %  args.dataset

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    train_dataset = build_dataset(args.dataset, DATA_DIR, train=True)
    test_dataset = build_dataset(args.dataset, DATA_DIR, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    n_channels, image_size, image_size = DATA_SHAPE[args.dataset]
    model = Classifier(n_channels, image_size, DATA_LABEL_NUM[args.dataset], 
                        n_filters=args.n_filters, hidden_dim=args.hidden_dim, 
                        conv=args.conv, label_dist=DATA_LABEL_DIST[args.dataset])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        for batch_idx, (data, label) in enumerate(train_loader):
            batch_size = len(data)
            data = data.to(device)
            label = label.to(device)

            out = model(data)
            if DATA_LABEL_DIST[args.dataset] == 'bernoulli':
                loss = F.binary_cross_entropy(out, label)
            elif DATA_LABEL_DIST[args.dataset] == 'categorical':
                loss = F.nll_loss(out, label)
            
            loss_meter.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))

        print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg, accuracy_meter.avg


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            for data, label in test_loader:
                batch_size = len(data)
                data = data.to(device)
                label = label.to(device)

                out = model(data)
                if DATA_LABEL_DIST[args.dataset] == 'bernoulli':
                    loss = F.binary_cross_entropy(out, label)
                elif DATA_LABEL_DIST[args.dataset] == 'categorical':
                    loss = F.nll_loss(out, label)

                loss_meter.update(loss.item(), batch_size)
                pbar.update()
            pbar.close()
        
        print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg, accuracy_meter.avg


    track_loss = np.zeros((args.epochs, 2))
    track_accuracy = np.zeros((args.epochs, 2))
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)

        track_loss[epoch - 1, 0] = train_loss
        track_loss[epoch - 1, 1] = test_loss
        track_accuracy[epoch - 1, 0] = train_acc
        track_accuracy[epoch - 1, 1] = test_acc
        
        np.save(os.path.join(args.out_dir, 'loss.npy'), track_loss)
        np.save(os.path.join(args.out_dir, 'accuracy.npy'), track_accuracy)
