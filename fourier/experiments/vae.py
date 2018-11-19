r"""Train a Variational Autoencoder."""

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

from .utils import elbo, AverageMeter, merge_args_with_dict
from .datasets import build_dataset
from .models import VAE
from .config import CONFIG
from . import DATA_OPTIONS, DATA_SHAPE, DATA_DIR, DATA_DIST


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='DynamicMNIST|PerturbMNIST|FashionMNIST|Histopathology|CelebA|SVHN|CIFAR10')
    parser.add_argument('conv', type=str, help='vanilla|coord')
    parser.add_argument('out_dir', type=str, help='where to save trained model')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    merge_args_with_dict(args, CONFIG)
    assert args.dataset in DATA_OPTIONS, "--dataset %s not recognized." %  args.dataset

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
    model = VAE(n_channels, image_size, args.z_dim, n_filters=args.n_filters,
                conv=args.conv, dist=DATA_DIST[args.dataset])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = len(data)
            data = data.to(device)

            out = model(data)
            loss = elbo(out, dist=DATA_DIST[args.dataset])
            loss_meter.update(-loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tELBO: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))

        print('====> Train Epoch: {}\tELBO: {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()

        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            for data, _ in test_loader:
                batch_size = len(data)
                data = data.to(device)
                log_p = model.get_marginal(data, n_samples=100)
                loss_meter.update(log_p.item(), batch_size)
                pbar.update()
            pbar.close()
        
        print('====> Test Epoch: {}\tlog p(x): {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg


    track_loss = np.zeros(args.epochs)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        scheduler.step()  # decrease learning rate
        test_loss = test(epoch)

        track_loss[epoch - 1] = test_loss
        np.save(os.path.join(args.out_dir, 'loss.npy'), track_loss)
