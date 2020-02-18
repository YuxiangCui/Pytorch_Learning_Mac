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


# num*len
# split
# 按照长度进行分割
# 前面标注分割的结果，以及对应的变量名，后面写清分割的方案或者片段长度
a = torch.rand(32, 8)
b = torch.rand_like(a)
e = torch.rand_like(a)

c = torch.stack([a, b], dim=0)
d = torch.stack([a, b, e], dim=0)
print(c.shape)
print(d.shape)

aa, bb = c.split([1, 1], dim=0)
print(aa.shape)
print(bb.shape)

aa, bb = d.split([2, 1], dim=0)
print(aa.shape)
print(bb.shape)

aa, bb = c.split(1, dim=0)
print(aa.shape)
print(bb.shape)

# chunk
# 按照分段数目进行分割

aa, bb = c.chunk(2, dim=0)
print(aa.shape)
print(bb.shape)