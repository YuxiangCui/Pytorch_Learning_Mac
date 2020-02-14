import torch
import numpy as np

# import from numpy
a = np.array([2.2,3.3])
print(torch.from_numpy(a))

a = np.ones([2,3])
print(torch.from_numpy(a))

# import from list
a = torch.tensor([2.2, 3.3])
print(a)

# torch.Tensor & torch.FloatTensor 可以接收shape而不只是现成的数据
a = torch.FloatTensor(2, 3)  # 自己初始化了一个2*3的tensor
print(a)

a = torch.FloatTensor([2, 3])
print(a)

a = torch.tensor([[2, 3], [4, 5]])
print(a)

# Uninitialized 一定注意后续数值的写入，否则为随机值

a = torch.empty(1)
print(a)

a = torch.Tensor(2, 3)
print(a)

a = torch.IntTensor([2, 3])
print(a)

a = torch.FloatTensor([2, 3])
print(a)

# Set default type 强化学习会采用DoubleTensor来提高精度，可以通过更改默认变量类型设定

a = torch.tensor([2, 3])
print(a.type())

a = torch.tensor([2.1, 3])
print(a.type())

torch.set_default_tensor_type(torch.DoubleTensor)
a = torch.tensor([2.1, 3])
print(a.type())

# rand/rand_like, randint  随机初始化

a = torch.rand(3, 3)  # 0-1均匀分布
print(a)

b = torch.rand_like(a)
print(b)

c = torch.randint(1, 2, [3, 3])  # low <= rand < high
print(c)

d = torch.randn(3,3)  # 正态分布 N（0，1）
print(d)

e = torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
print(e)


# full  size, value

a = torch.full([2, 3], 7)
print(a)

a = torch.full([], 7)  # scalar
print(a)

a = torch.full([1], 7)  # vector
print(a)

# arange/range  start, end, step

a = torch.arange(0, 10)
print(a)

a = torch.arange(0, 10, 2)
print(a)

# a = torch.range(0, 10)  不建议使用
# print(a)


# linspace/logspace  start, end,
a = torch.linspace(0, 10, 3)
print(a)

a = torch.linspace(0, 10, 5)
print(a)

a = torch.linspace(0, 10, 7)
print(a)

a = torch.logspace(0, 10, 11, 2)  # start, end, steps, base
print(a)

a = torch.logspace(0, 10, 11, np.e)  # start, end, steps, base
print(a)

a = torch.logspace(0, 10, 11, 10)  # start, end, steps, base
print(a)

# ones/zeros/eye 1,0,对角

a = torch.ones(3, 3)
print(a)

b = torch.zeros(3, 3)
print(b)

c = torch.eye(3,4)
print(c)

d = torch.eye(3, 3)
print(d)

e = torch.eye(3)
print(e)

f = torch.zeros(3,3)
g = torch.ones_like(f)
print(g)

# randperm 随机打散 random + shuffle

a = torch.randperm(10)
print(a)

# b = torch.randperm(3, 3)
# print(b) 这样用不对，只能生成一个随机数组，不包含提供的最大值

b = torch.rand(2, 3)
c = torch.rand(2, 2)
idx = torch.randperm(2)
print(b)
print(c)

b = b[idx]  # shuffle 按照行的顺序
c = c[idx]
print(b)
print(c)

print(b[[0, 1]])
print(b[[1, 0]])

