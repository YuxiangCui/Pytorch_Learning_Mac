# pytorch_learning_12_perceptron

import torch
from torch.nn import functional as F

# 多输入单输出
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)

o = torch.sigmoid(x@w.t())
print(o)  # tensor([[0.4561]], grad_fn=<SigmoidBackward>)
print(o.shape)  # torch.Size([1, 1])

loss = F.mse_loss(torch.ones(1, 1), o)
print(loss.shape)  # torch.Size([])

loss.backward()
print(w.grad)
# tensor([[ 0.3117, -0.2556,  0.0821,  0.3212, -0.1595,  0.1250,  0.0498,  0.0400,
#          -0.1263,  0.0531]])
print("----------------------------")


# 多输入多输出
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
o = torch.sigmoid(x@w.t())

print(o.shape)
# torch.Size([1, 2])
loss = F.mse_loss(torch.ones(1, 2), o)  # (1, 1)也可以，自动broadcast
print(loss)
# tensor(0.4635, grad_fn=<MeanBackward0>)
loss.backward()

print(w.grad)
# tensor([[ 0.0238,  0.0177,  0.0504, -0.0008,  0.0171,  0.0481, -0.0065,  0.0161,
#          -0.0221, -0.0661],
#         [ 0.0387,  0.0287,  0.0818, -0.0013,  0.0278,  0.0780, -0.0105,  0.0261,
#          -0.0359, -0.1073]])
