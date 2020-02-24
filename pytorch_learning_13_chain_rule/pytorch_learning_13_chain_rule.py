# pytorch_learning_13_chain_rule

import torch

# chain rule
# dy / dx = dy / du * du / dx
# 中间变量，隐藏层参数


# y_1 = x * w_1 + b_1
# y_2 = y_1 * w_2 + b_2

x = torch.tensor(1.)
w_1 = torch.tensor(2., requires_grad=True)
b_1 = torch.tensor(1.)
w_2 = torch.tensor(2., requires_grad=True)
b_2 = torch.tensor(1.)

y_1 = x * w_1 + b_1
y_2 = y_1 * w_2 + b_2

dy2_dy1 = torch.autograd.grad(y_2, [y_1], retain_graph=True)[0]
dy1_dw1 = torch.autograd.grad(y_1, [w_1], retain_graph=True)[0]
dy2_dw1 = torch.autograd.grad(y_2, [y_1], retain_graph=True)[0]

print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)
# tensor(2.)

