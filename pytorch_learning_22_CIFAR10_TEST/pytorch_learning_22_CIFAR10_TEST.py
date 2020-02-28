# pytorch_learning_22_CIFAR10_TEST
# ResNet

# CIFAR-10
# airplane automoblie bird cat deer dog frog horse ship truck
# 50000 training 10000 test
# 32 x 32

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def main():
    batchsize = 32
    cifar_train = datasets.CIFAR10('../pytorch_learning_dataset/CIFAR', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_test = datasets.CIFAR10('../pytorch_learning_dataset/CIFAR', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsize, shuffle=True)
    x, label = iter(cifar_train).next()
    print('x: ', x.shape, 'label: ', label.shape)


if __name__ == '__main__':
    main()
