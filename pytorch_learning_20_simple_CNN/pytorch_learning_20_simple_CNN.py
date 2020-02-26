# pytorch_learning_20_simple_CNN

import torch
import torch.nn as nn
from torch.nn import functional as F

# 卷积层

# 卷积 convolution
# image + kernel => feature map
# * input channels 输入通道，e.g. RGB
# * kernel channels 多个不同功能的卷积核，与bias大小相同
# * kernel size 核大小
# * stride
# * padding
# kernel 滑动窗口
# 根据kernel的设定，存在不同的功能，边缘检测，模糊等
# 多个kernel，就生成多个内容不同的feature map
# padding 操作 在image周围补像素，使得卷积后图像大小不变


# nn.Conv2d
# input, kernel channels
layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)
out = layer(x)
print(out.shape)  # torch.Size([1, 3, 26, 26])

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
x = torch.rand(1, 1, 28, 28)
out = layer(x)
print(out.shape)   # torch.Size([1, 3, 28, 28])

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
x = torch.rand(1, 1, 28, 28)
out = layer(x)
print(out.shape)     # torch.Size([1, 3, 14, 14])

print(layer.weight)
print(layer.weight.shape)  # torch.Size([3, 1, 3, 3])
print(layer.bias.shape)  # torch.Size([3])

print("------------------------------------------")

# F.conv2d
x = torch.randn(1, 3, 28, 28)
w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)

out = F.conv2d(x, w, b, stride=1, padding=1)
print(out.shape)  # torch.Size([1, 16, 26, 26])

print("------------------------------------------")


# 池化层 pooling

# DOWN / UP SAMPLE
# * pooling 池化
# * upsample 上采样
# * ReLU
# pooling 池化— 缩小图片
# max pooling  / average pooling
# stride 步长 滑动窗口类似于卷积
# subsampling  隔行采样隔列采样

x = out

layer = nn.MaxPool2d(2, stride=2)
layer = nn.AvgPool2d(2, stride=2)

# or
out_1 = F.avg_pool2d(x, 2, stride=2)
print(out_1)
print(layer(x).shape)  # torch.Size([1, 16, 13, 13])
print("------------------------------------------")


# upsample 上采样

x = out
out = F.interpolate(x, scale_factor=2, mode='nearest')
print(out.shape)  # torch.Size([1, 16, 52, 52])
out = F.interpolate(x, scale_factor=3, mode='nearest')
print(out.shape)  # torch.Size([1, 16, 78, 78])

print("------------------------------------------")

# ReLU
x = out
layer = nn.ReLU(inplace=True)  # 在原内存空间上进行计算
out = layer(x)
print(out.shape)

out = F.relu(x)
print(out.shape)

# Batch Normalization
# 统计当前batch的均值和方差，将原来分布化到0附近高斯分布，方差和均值较小，beta gamma 偏置 也即 bias 和 weight
# running 记录更新
# beta gamma 偏置持续更新
# 每一个batch100张图片，计算每个channel上的均值方差
x = torch.rand(100, 16, 784)
layer = nn.BatchNorm1d(16)  # 参数为channel数目
out = layer(x)

print(layer.running_mean)
print(layer.running_var)


# test之前要把参数改回，layer.eval()
# model.train() ：启用 BatchNormalization 和 Dropout
# model.eval() ：不启用 BatchNormalization 和 Dropout
