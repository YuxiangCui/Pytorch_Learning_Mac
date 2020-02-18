# add/minus/multiply/divide
# matmul
# pow
# sqrt/rsqrt
# round

import torch

a = torch.rand(3, 4)
b = torch.rand(4)
# broadcast机制会自动进行调整
print(a+b)
