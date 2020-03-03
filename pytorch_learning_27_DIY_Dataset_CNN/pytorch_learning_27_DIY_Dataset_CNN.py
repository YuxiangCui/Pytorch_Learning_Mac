# pytorch_laerning_27_DIY_Dataset_CNN

# Load data
# Build model
# Train and Test
# Transfer learning

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

import torchvision
from torchvision import transforms as transforms
from matplotlib import pyplot as plt



# Load data
# Inherit from torch.utils.data.Dataset
# __init__ 继承自初始化类，根据标志确定训练集，验证集，测试集
# __len__  返回数据量
# __getitem__ 根据索引返回当前位置元素

# Image Resize  224*224 for ResNet18
# Data Argumentation  e.g. Rotate Crop
# Normalize  Mean&std
# ToTensor

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1,shape)


def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{} :  {}".format(name, label[i].item()))
        plt.xticks()
        plt.yticks()
    plt.show()



class ResidualBlock(nn.Module):
    """
    ResNet Block
    """
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.ResBlock = nn.Sequential(
            # 通过增大stride可以减少channel数目
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            # [b, in_channel, h, w] => [b, out_channel, h, w]
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.ResBlock(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet, self).__init__()
        self.in_channel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.build_layer(ResidualBlock, 32,  2, stride=3)
        self.layer2 = self.build_layer(ResidualBlock, 64, 2, stride=3)
        self.layer3 = self.build_layer(ResidualBlock, 128, 2, stride=2)
        self.layer4 = self.build_layer(ResidualBlock, 256, 2, stride=2)  # 一般到这个维度差不多了
        self.fc = nn.Linear(256*3*3, num_classes)

    def build_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        # print('before pool', out.shape)
        # out = F.avg_pool2d(out, 8)
        # out = F.adaptive_avg_pool2d(out, [1, 1])
        # print('after pool', out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)

def main():
    # test 检查维度是否相同
    blk = ResidualBlock(64, 128)
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    print("block: ", out.shape)

    x = torch.randn(2, 3, 224, 224)
    model = ResNet()
    out = model(x)
    print("ResNet: ", out.shape)
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size: ',p)


if __name__ == '__main__':
    main()
