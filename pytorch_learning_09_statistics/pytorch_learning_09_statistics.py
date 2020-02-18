# pytorch_learning_09_statistics
# norm
# mean sum
# prod
# max, min, argmin, argmax
# kthvalue, topk

import torch

# norm != normalize
# norm表示范数，normalize表示正则化，例如batch_norm

# matrix norm v.s. vector norm 矩阵范数 & 向量范数  这里都考虑元素的范数
a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)

print(a.norm(1), b.norm(1), c.norm(1))  # tensor(8.) tensor(8.) tensor(8.)

print(a.norm(2), b.norm(2), c.norm(2))  # tensor(2.8284) tensor(2.8284) tensor(2.8284)

print(b.norm(1, dim=1))  # tensor([4., 4.])

print(b.norm(2, dim=1))  # tensor([2., 2.])

print(c.norm(1, dim=0))
# tensor([[2., 2.],
#         [2., 2.]])

print(c.norm(2, dim=0))
# tensor([[1.4142, 1.4142],
#         [1.4142, 1.4142]])


# mean, sum, prod, max, min, argmin, argmax

a = torch.arange(8).view(2, 4).float()
print(a.min(), a.max(), a.mean(), a.prod(), a.sum(), a.argmax(), a.argmin())
# tensor(0.) tensor(7.) tensor(3.5000) tensor(0.) tensor(28.) tensor(7) tensor(0)
# 注意这里argmax和argmin如果不给定具体的维度，会把原矩阵展平然后排序
a = torch.randn(4, 10)
print(a.argmax())  # tensor(37)
print(a.argmax(dim=1))  # tensor([7, 1, 0, 7])

# dim 操作所在维度
# keepdim 保持原有维度
print(a)
print(a.shape)  # torch.Size([4, 10])
print(a.max(dim=1))
# torch.return_types.max(
# values=tensor([1.8498, 0.9763, 1.5220, 1.5875]),
# indices=tensor([4, 0, 5, 9]))
print(a.argmax(dim=0))  # tensor([1, 1, 1, 1, 0, 2, 2, 0, 0, 3])
print(a.argmax(dim=1))  # tensor([4, 0, 5, 9])
print(a.max(dim=1, keepdim=True))
# torch.return_types.max(
# values=tensor([[1.8498],
#         [0.9763],
#         [1.5220],
#         [1.5875]]),
# indices=tensor([[4],
#         [0],
#         [5],
#         [9]]))
print(a.argmax(dim=1, keepdim=True))
# tensor([[4],
#         [0],
#         [5],
#         [9]])


# top-k k-th
# topk()  kthvalue()


a = torch.rand(4,10)
print(a)
# tensor([[0.6791, 0.7902, 0.5159, 0.6750, 0.7883, 0.2556, 0.8518, 0.3396, 0.4248,
#          0.3081],
#         [0.5667, 0.2990, 0.0226, 0.1047, 0.9171, 0.3075, 0.4297, 0.2149, 0.2353,
#          0.6848],
#         [0.4322, 0.6734, 0.8619, 0.2445, 0.1153, 0.4752, 0.6293, 0.9444, 0.5243,
#          0.3969],
#         [0.3275, 0.2683, 0.7767, 0.8909, 0.7604, 0.5456, 0.8559, 0.1823, 0.3995,
#          0.1417]])

print(a.topk(3,dim=1))
# torch.return_types.topk(
# values=tensor([[0.8518, 0.7902, 0.7883],
#         [0.9171, 0.6848, 0.5667],
#         [0.9444, 0.8619, 0.6734],
#         [0.8909, 0.8559, 0.7767]]),
# indices=tensor([[6, 1, 4],
#         [4, 9, 0],
#         [7, 2, 1],
#         [3, 6, 2]]))
print(a.topk(3,dim=1,largest=False))  # 找最小的
# torch.return_types.topk(
# values=tensor([[0.2556, 0.3081, 0.3396],
#         [0.0226, 0.1047, 0.2149],
#         [0.1153, 0.2445, 0.3969],
#         [0.1417, 0.1823, 0.2683]]),
# indices=tensor([[5, 9, 7],
#         [2, 3, 7],
#         [4, 3, 9],
#         [9, 7, 1]]))
# 找第n小的
print(a.kthvalue(8, dim=1))  # 这里第8小就是第3大
# torch.return_types.kthvalue(
# values=tensor([0.7883, 0.5667, 0.6734, 0.7767]),
# indices=tensor([4, 0, 1, 2]))
print(a.kthvalue(3))
# torch.return_types.kthvalue(
# values=tensor([0.3396, 0.2149, 0.3969, 0.2683]),
# indices=tensor([7, 7, 9, 1]))
print(a.kthvalue(3, dim=1))
# torch.return_types.kthvalue(
# values=tensor([0.3396, 0.2149, 0.3969, 0.2683]),
# indices=tensor([7, 7, 9, 1]))


# >, <, >=, <=, !=, ==
# element wise
a = torch.rand(4,10)
print(a)
# tensor([[0.6267, 0.2253, 0.7594, 0.8550, 0.2979, 0.5881, 0.0021, 0.6846, 0.1417,
#          0.7474],
#         [0.9613, 0.0323, 0.0514, 0.9293, 0.7174, 0.1316, 0.7375, 0.9503, 0.0238,
#          0.1456],
#         [0.6618, 0.0177, 0.1503, 0.8186, 0.2548, 0.7577, 0.1693, 0.7651, 0.1289,
#          0.9982],
#         [0.5212, 0.8210, 0.7766, 0.2506, 0.2080, 0.9902, 0.7905, 0.5490, 0.6631,
#          0.2815]])
print(a > 0.5)
print(torch.gt(a, 0.5))
# tensor([[ True, False,  True,  True, False,  True, False,  True, False,  True],
#         [ True, False, False,  True,  True, False,  True,  True, False, False],
#         [ True, False, False,  True, False,  True, False,  True, False,  True],
#         [ True,  True,  True, False, False,  True,  True,  True,  True, False]])
print(a != 0)
# tensor([[True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True]])

a = torch.ones(2, 3)
b = torch.zeros_like(a)

print(torch.eq(a, b))
# tensor([[False, False, False],
#         [False, False, False]])
print(torch.eq(a, a))
# tensor([[True, True, True],
#         [True, True, True]])
print(torch.equal(a, a))  # True


