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
from torch import nn
from lenet5 import Lenet_5


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

    # GPU
    # device = torch.device('cuda')
    # model = Lenet_5().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)

    criterion = nn.CrossEntropyLoss()
    model = Lenet_5()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):
        model.train()
        for batch_idx, (x, label) in enumerate(cifar_train):
            # x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('TRAIN', epoch, loss.item())  # 最后的一个batch的loss

        model.eval()
        with torch.no_grad():  # 后面不需要求梯度
            num_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # x, label = x.to(device), label.to(device)
                logits = model(x)
                predict = logits.argmax(dim=1)

                num_correct += torch.eq(predict, label).float().sum().item()  # 当前batch中正确的数目
                total_num += x.size(0)

            accuracy = num_correct / total_num
            print('TEST', epoch, accuracy)





if __name__ == '__main__':
    main()
