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



