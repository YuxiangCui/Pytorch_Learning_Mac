import torch
a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)