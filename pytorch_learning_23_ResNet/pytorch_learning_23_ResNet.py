import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.build_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.build_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.build_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.build_layer(ResidualBlock, 512, 2, stride=1)  # 一般到这个维度差不多了
        self.fc = nn.Linear(512, num_classes)

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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print('before pool', out.shape)
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, [1, 1])
        # print('after pool', out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)

#
# def main():
#     # test 检查维度是否相同
#     blk = ResidualBlock(64, 128, stride=2)
#     tmp = torch.randn(2, 64, 32, 32)
#     out = blk(tmp)
#     print("block: ", out.shape)
#
#     x = torch.randn(2, 3, 32, 32)
#     model = ResNet()
#     out = model(x)
#     print("ResNet: ", out.shape)
#
#
# if __name__ == '__main__':
#     main()