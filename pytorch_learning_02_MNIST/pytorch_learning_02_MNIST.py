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

# x, y = next(iter(train_loader))
# print(x.shape, y.shape, x.min(), x.max())
# plot_image(x, y, 'image_sample')


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


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


train_loss = []
for epoch in range (3):

    for batch_idx, (x, y) in enumerate(train_loader):

        # 先展开为对应的网络输入格式，维度
        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784] =>[b, 10]
        x = x.view(x.size(0), 28*28)
        out = net(x)
        y_onehot = one_hot(y)

        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()
        loss.backward()
        # w' = w - lr * grad
        optimizer.step()

        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())
plot_figure(train_loss)


# TEST
total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    # out: [batch_size, 10] 获取dim=1维度上最大的值对应的索引
    pred = out.argmax(dim=1)
    # 当前batch中正确的个数，继而进行统计
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
accuracy = total_correct / total_num
print("test accuracy: ", accuracy)

x, y = next(iter(test_loader))
out = net(x = x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, "test sample")