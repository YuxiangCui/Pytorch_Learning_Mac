import torch
import numpy as np

# import from numpy
a = np.array([2.2,3.3])
print(torch.from_numpy(a))  # tensor([2.2000, 3.3000], dtype=torch.float64)

a = np.ones([2,3])
print(torch.from_numpy(a))
# tensor([[1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)

# import from list
a = torch.tensor([2.2, 3.3])
print(a)  # tensor([2.2000, 3.3000])

# torch.Tensor & torch.FloatTensor 可以接收shape而不只是现成的数据
a = torch.FloatTensor(2, 3)  # 自己初始化了一个2*3的tensor
print(a)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])


a = torch.FloatTensor([2, 3])
print(a)  # tensor([2., 3.])


a = torch.tensor([[2, 3], [4, 5]])
print(a)
# tensor([[2, 3],
#         [4, 5]])

# Uninitialized 一定注意后续数值的写入，否则为随机值

a = torch.empty(1)
print(a)  # tensor([7.0065e-45])

a = torch.Tensor(2, 3)
print(a)
# tensor([[ 0.0000e+00,  5.8413e-06, -6.1208e+23],
#         [ 2.7837e-40,  4.9612e-30,  1.4013e-45]])

a = torch.IntTensor([2, 3])
print(a)
# tensor([2, 3], dtype=torch.int32)

a = torch.FloatTensor([2, 3])
print(a)
# tensor([2., 3.])

# Set default type 强化学习会采用DoubleTensor来提高精度，可以通过更改默认变量类型设定

a = torch.tensor([2, 3])
print(a.type())  # torch.LongTensor

a = torch.tensor([2.1, 3])
print(a.type())  # torch.FloatTensor

torch.set_default_tensor_type(torch.DoubleTensor)
a = torch.tensor([2.1, 3])
print(a.type())  # torch.DoubleTensor

# rand/rand_like, randint  随机初始化

a = torch.rand(3, 3)  # 0-1均匀分布
print(a)
# tensor([[0.0090, 0.2389, 0.1770],
#         [0.8717, 0.2723, 0.6541],
#         [0.0187, 0.8649, 0.3536]])

b = torch.rand_like(a)
print(b)
# tensor([[0.7082, 0.1881, 0.5463],
#         [0.4117, 0.0362, 0.3035],
#         [0.4009, 0.4509, 0.7530]])


c = torch.randint(1, 4, [3, 3])  # low <= rand < high
print(c)
# tensor([[1, 1, 1],
#         [2, 2, 3],
#         [3, 1, 3]])

d = torch.randn(3,3)  # 正态分布 N（0，1）
print(d)
# tensor([[-0.4998, -2.2904,  0.2071],
#         [ 0.5809, -1.7752,  0.4195],
#         [ 0.6013, -0.3266, -0.0888]])

e = torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
print(e)
# tensor([-0.3113,  0.5304, -0.4583,  0.7140, -0.1470, -0.1260,  0.5031, -0.4727, -0.0888,  0.0833])

# full  size, value

a = torch.full([2, 3], 7)
print(a)
# tensor([[7., 7., 7.],
#         [7., 7., 7.]])

a = torch.full([], 7)  # scalar
print(a)  # tensor(7.)

a = torch.full([1], 7)  # vector
print(a)  # tensor([7.])

# arange/range  start, end, step

a = torch.arange(0, 10)
print(a)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

a = torch.arange(0, 10, 2)
print(a)  # tensor([0, 2, 4, 6, 8])

# a = torch.range(0, 10)  不建议使用
# print(a)


# linspace/logspace  start, end,
a = torch.linspace(0, 10, 3)
print(a)  # tensor([ 0.,  5., 10.])

a = torch.linspace(0, 10, 5)
print(a)  # tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])

a = torch.linspace(0, 10, 7)
print(a)  # tensor([ 0.0000,  1.6667,  3.3333,  5.0000,  6.6667,  8.3333, 10.0000])

a = torch.logspace(0, 10, 11, 2)  # start, end, steps, base
print(a)
# tensor([1.0000e+00, 2.0000e+00, 4.0000e+00, 8.0000e+00, 1.6000e+01, 3.2000e+01,
#         6.4000e+01, 1.2800e+02, 2.5600e+02, 5.1200e+02, 1.0240e+03])

a = torch.logspace(0, 10, 11, np.e)  # start, end, steps, base
print(a)
# tensor([1.0000e+00, 2.7183e+00, 7.3891e+00, 2.0086e+01, 5.4598e+01, 1.4841e+02,
#         4.0343e+02, 1.0966e+03, 2.9810e+03, 8.1031e+03, 2.2026e+04])

a = torch.logspace(0, 10, 11, 10)  # start, end, steps, base
print(a)
# tensor([1.0000e+00, 1.0000e+01, 1.0000e+02, 1.0000e+03, 1.0000e+04, 1.0000e+05,
#         1.0000e+06, 1.0000e+07, 1.0000e+08, 1.0000e+09, 1.0000e+10])

# ones/zeros/eye 1,0,对角

a = torch.ones(3, 3)
print(a)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])

b = torch.zeros(3, 3)
print(b)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])

c = torch.eye(3,4)
print(c)
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.]])

d = torch.eye(3, 3)
print(d)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

e = torch.eye(3)
print(e)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

f = torch.zeros(3,3)
g = torch.ones_like(f)
print(g)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])

# randperm 随机打散 random + shuffle

a = torch.randperm(10)
print(a)  # tensor([6, 4, 1, 2, 7, 0, 3, 5, 9, 8])

# b = torch.randperm(3, 3)
# print(b) 这样用不对，只能生成一个随机数组，不包含提供的最大值

b = torch.rand(2, 3)
c = torch.rand(2, 2)
idx = torch.randperm(2)
print(b)
# tensor([[0.4055, 0.9506, 0.2289],
#         [0.0939, 0.6379, 0.5605]])
print(c)
# tensor([[0.2927, 0.4415],
#         [0.3089, 0.5034]])

b = b[idx]  # shuffle 按照行的顺序
c = c[idx]
print(b)
# tensor([[0.4055, 0.9506, 0.2289],
#         [0.0939, 0.6379, 0.5605]])
print(c)
# tensor([[0.2927, 0.4415],
#         [0.3089, 0.5034]])


print(b[[0, 1]])
# tensor([[0.4055, 0.9506, 0.2289],
#         [0.0939, 0.6379, 0.5605]])

print(b[[1, 0]])
# tensor([[0.0939, 0.6379, 0.5605],
#         [0.4055, 0.9506, 0.2289]])


