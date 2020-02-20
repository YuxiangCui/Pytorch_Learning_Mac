# pytorch_learning_11_gradient
# loss and activation function
import torch
from torch.nn import functional as F

a = torch.linspace(-100,100,10)
print(a)
# tensor([-100.0000,  -77.7778,  -55.5556,  -33.3333,  -11.1111,   11.1111,
#           33.3333,   55.5555,   77.7778,  100.0000])
print(torch.sigmoid(a))
# tensor([0.0000e+00, 1.6655e-34, 7.4564e-25, 3.3382e-15, 1.4945e-05, 9.9999e-01,
#         1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00])

a = torch.linspace(-1, 1, 10)
print(a)
# tensor([-1.0000, -0.7778, -0.5556, -0.3333, -0.1111,  0.1111,  0.3333,  0.5556,
#          0.7778,  1.0000])
print(torch.tanh(a))
# tensor([-0.7616, -0.6514, -0.5047, -0.3215, -0.1107,  0.1107,  0.3215,  0.5047,
#          0.6514,  0.7616])
print(torch.relu(a))
# tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1111, 0.3333, 0.5556, 0.7778,
#         1.0000])