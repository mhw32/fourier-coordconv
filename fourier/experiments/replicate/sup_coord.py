from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from ..models import CoordConv2d


data_dir = '/mnt/fs5/wumike/datasets/data-quadrant'
train_data = np.load(os.path.join(data_dir, 'train_set.npy'))
train_labels = np.load(os.path.join(data_dir, 'train_onehot.npy'))
test_data = np.load(os.path.join(data_dir, 'test_set.npy'))
test_labels = np.load(os.path.join(data_dir, 'test_onehot.npy'))

train_labels = train_labels.reshape((-1, 64 * 64)).astype('int64')
test_labels = test_labels.reshape((-1, 64 * 64)).astype('int64')

train_tensor_x = torch.stack([torch.Tensor(i) for i in train_set])
train_tensor_y = torch.stack([torch.LongTensor(i) for i in train_onehot])

train_dataset = utils.TensorDataset(train_tensor_x,train_tensor_y)
train_dataloader = utils.DataLoader(train_dataset, batch_size=32, shuffle=False)

test_tensor_x = torch.stack([torch.Tensor(i) for i in test_set])
test_tensor_y = torch.stack([torch.LongTensor(i) for i in test_onehot])

test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y)
test_dataloader = utils.DataLoader(test_dataset, batch_size=32, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.coordconv = CoordConv2d(2, 32, 1, with_r=True)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def train(epoch):
    net.train()
    iters = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = cross_entropy_one_hot(output, target)
        loss.backward()
        optimizer.step()
        iters += len(data)
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, iters, len(train_dataloader.dataset),
                100. * (batch_idx + 1) / len(train_dataloader), loss.item()), end='\r', flush=True)
    print("")


def test():
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_dataloader:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += cross_entropy_one_hot(output, target).item()
            _, pred = output.max(1, keepdim=True)
            _, label = target.max(dim=1)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss = test_loss
    test_loss /= len(test_dataloader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


epochs = 10
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
