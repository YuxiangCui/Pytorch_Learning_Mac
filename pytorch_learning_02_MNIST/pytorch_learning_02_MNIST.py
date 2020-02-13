# TRAIN
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from torchvision import transforms as transforms
from matplotlib import pyplot as plt

from utils import plot_figure, plot_image, one_hot

batch_size = 512


# Load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        'mnist_data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        'mnist_data/',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size,
    shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image_sample')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

