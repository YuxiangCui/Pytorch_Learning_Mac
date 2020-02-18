# cat
# stack
# split
# chunk

import torch


# cat(list, dim)
# 确保除了合并的维度外的所有维度的shape相同
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
c = torch.cat([a, b], dim=0)
print(c.shape)  # torch.Size([9, 32, 8])

# stack(list, dim) 但是与cat不同在于，stack会创建新的维度，表示合并时的单元对应的新的维度，因此需要合并前的维度完全一样
a = torch.rand(4, 3, 16, 16)
b = torch.rand(4, 3, 16, 16)
b_0 = torch.rand(3, 3, 16, 16)
c = torch.cat([a, b], dim=0)  # torch.Size([8, 3, 16, 16])
d = torch.stack([a, b], dim=0)  # torch.Size([2, 4, 3, 16, 16])

print(c.shape)  # torch.Size([8, 3, 16, 16])
print(d.shape)  # torch.Size([2, 4, 3, 16, 16])
# print(torch.stack([a, b_0], dim=0))  BUG:stack对应的元素必须相同



