import torch
import numpy as np

a = np.array([2.2,3.3])
print(torch.from_numpy(a))

a = np.ones([2,3])
print(torch.from_numpy(a))