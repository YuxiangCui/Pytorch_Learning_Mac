# lenet5

import torch
from torch import nn
from torch.nn import functional as F

class Lenet_5(nn.Module):
    """
    CIFAR-10 train
    """
    def __init__(self):
        super(Lenet_5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        # flatten 压平操作

        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 10)
        )

        self.critierion = nn.CrossEntropyLoss()  # 这里包含了softmax

    def forward(self, x):
        """
        :param x: [batch_size, 3, 32, 32]
        :return:
        """
        batch_size = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batch_size, 16*5*5)
        logits = self.fc_unit(x)

        return logits




