#!/usr/bin/env python 
""" Replication of CoordConv Paper.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import torch.utils.data as utils
import torch.nn.functional as F

from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from ..models import CoordConv2d as OurCoordConv2d

# Simple CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.coordconv = CoordConv2d(2, 32, 1, with_r=True)
        self.coordconv = OurCoordConv2d(2, 32, 1, with_r=True)
        self.conv1 = nn.Conv2d(32, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64,  1, 1)
        self.conv4 = nn.Conv2d( 1,  1, 1)

    def forward(self, x):
        x = self.coordconv(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(-1, 64*64)
        return x

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

# Add Coordinates to Tensor
class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        
        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]
            
            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)
            
            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)
            
            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +\
                                torch.pow(yy_channel - 0.5, 2) +\
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out

# Coordinate Convolution
class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(in_channels+self.rank+int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        out = self.addcoords(input_tensor)
        out = self.conv(out)
        return out

# Dataset Loading
def load_dataset(datatype):
    if datatype == 'uniform':
        # Load the one hot datasets
        train_onehot = np.load('./data-uniform/train_onehot.npy').astype('float32')
        test_onehot = np.load('./data-uniform/test_onehot.npy').astype('float32')

        # make the train and test datasets
        # train
        pos_train = np.where(train_onehot == 1.0)
        X_train = pos_train[2]
        Y_train = pos_train[3]
        train_set = np.zeros((len(X_train), 2, 1, 1), dtype='float32')
        for i, (x, y) in enumerate(zip(X_train, Y_train)):
            train_set[i, 0, 0, 0] = x
            train_set[i, 1, 0, 0] = y

        # test
        pos_test = np.where(test_onehot == 1.0)
        X_test = pos_test[2]
        Y_test = pos_test[3]
        test_set = np.zeros((len(X_test), 2, 1, 1), dtype='float32')
        for i, (x, y) in enumerate(zip(X_test, Y_test)):
            test_set[i, 0, 0, 0] = x
            test_set[i, 1, 0, 0] = y

        train_set = np.tile(train_set, [1, 1, 64, 64])
        test_set = np.tile(test_set, [1, 1, 64, 64])

        # Normalize the datasets
        train_set /= (64. - 1.)  # 64x64 grid, 0-based index
        test_set /= (64. - 1.)  # 64x64 grid, 0-based index

        print('Train set : ', train_set.shape, train_set.max(), train_set.min())
        print('Test set : ', test_set.shape, test_set.max(), test_set.min())
        return train_set, test_set, train_onehot, test_onehot
    else:
        # Load the one hot datasets and the train / test set
        train_set = np.load('./data-quadrant/train_set.npy').astype('float32')
        test_set = np.load('./data-quadrant/test_set.npy').astype('float32')

        train_onehot = np.load('./data-quadrant/train_onehot.npy').astype('float32')
        test_onehot = np.load('./data-quadrant/test_onehot.npy').astype('float32')

        train_set = np.tile(train_set, [1, 1, 64, 64])
        test_set = np.tile(test_set, [1, 1, 64, 64])

        # Normalize datasets
        train_set /= train_set.max()
        test_set /= test_set.max()

        print('Train set : ', train_set.shape, train_set.max(), train_set.min())
        print('Test set : ', test_set.shape, test_set.max(), test_set.min())
        return train_set, test_set, train_onehot, test_onehot

def train(epoch, net, train_dataloader, optimizer, criterion, device):
    net.train()
    iters = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        iters += len(data)
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, iters, len(train_dataloader.dataset),
                100. * (batch_idx + 1) / len(train_dataloader), loss.data.item()))
    print("")


def test(net, test_loader, optimizer, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    pred_logits = torch.tensor([])
    import pdb; pdb.set_trace()
    for data, target in test_loader:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
        output = net(data)
        logits = F.softmax(output, dim=1)
        pred_logits = torch.cat((pred_logits, logits.cpu()), dim=0)
        test_loss += criterion(output, target).item()
        _, pred = output.max(1, keepdim=True)
        _, label = target.max(dim=1)
        correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    pred_logits = torch.sum(pred_logits, dim=0)

if __name__ == '__main__':
    # set seeds for reproducability
    np.random.seed(0)
    torch.manual_seed(0)

    # retrieve datasets
    datatype = 'uniform' #'quadrant'  # 
    assert datatype in ['uniform', 'quadrant']
    train_set, test_set, train_onehot, test_onehot = load_dataset(datatype)

    # flattent datasets
    train_onehot = train_onehot.reshape((-1, 64 * 64)).astype('int64')
    test_onehot = test_onehot.reshape((-1, 64 * 64)).astype('int64')

    # initialize network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    # train data
    train_tensor_x = torch.stack([torch.Tensor(i) for i in train_set])
    train_tensor_y = torch.stack([torch.LongTensor(i) for i in train_onehot])
    train_dataset = utils.TensorDataset(train_tensor_x,train_tensor_y)
    train_dataloader = utils.DataLoader(train_dataset, batch_size=32, shuffle=False)

    # test data 
    test_tensor_x = torch.stack([torch.Tensor(i) for i in test_set])
    test_tensor_y = torch.stack([torch.LongTensor(i) for i in test_onehot])
    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # train model
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = cross_entropy_one_hot
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epoch, net, train_dataloader, optimizer, criterion, device)

    # test model
    test(net, test_dataloader, optimizer, criterion, device)
