# lenet5

import torch
from torch import nn


class Lenet_5(nn.Module):
    """
    CIFAR-10 train
    """
    def __init__(self):
        super(Lenet_5, self).__init__()