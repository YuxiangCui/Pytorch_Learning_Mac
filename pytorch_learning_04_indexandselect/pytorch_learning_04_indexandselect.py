import torch
a = torch.rand(4, 3, 28, 28)

# 指定通道取出
print(a[0].shape)  # torch.Size([3, 28, 28])
print(a[0, 0].shape)  # torch.Size([28, 28])
print(a[0, 0, 2, 4])  # tensor(0.5278)
print("\n")

# 通用形式  start:end:step

# 连续取
# 一个:，表示取全部
# :加数字表示从0取到数字之前，:2表示0，1，不包含最后一个       选取部分所在的方向为 ==>
# 数字加:表示从数字取到最后，rgb，1:，表示gb，包含起始位       选取部分所在的方向为 <==
# a:b表示取区间内部，[a, b)，左开右闭
# 负数表示反向索引    [0, 1, 2]   [-3, -2, -1]
print(a[:2, :2].shape)  # torch.Size([2, 2, 28, 28])
print(a[:2, :2, :, :].shape)  # torch.Size([2, 2, 28, 28])
print(a[:2, 1:, :, :].shape)  # torch.Size([2, 2, 28, 28])
print(a[:2, 0:2, :, :].shape)  # torch.Size([2, 2, 28, 28])
print(a[:2, -1:, :, :].shape)  # torch.Size([2, 1, 28, 28])
print("\n")


# 间隔取
print(a[:, :, 0:28:2, 0:28:2].shape)  # torch.Size([4, 3, 14, 14])
print(a[:, :, ::2, ::2].shape)  # torch.Size([4, 3, 14, 14])
print("\n")

# 选取指定的位置
print(a.index_select(0, torch.tensor([0, 2, 3])).shape)  # 这里的参数为指定的位置，第023张图片
# torch.Size([3, 3, 28, 28])
print(a.index_select(1, torch.tensor([1, 2])).shape)  # 这里的参数为指定的位置，GB通道
# torch.Size([4, 2, 28, 28])
print(a.index_select(2, torch.arange(28)).shape)  # 取前28行
# torch.Size([4, 3, 28, 28])
print(a.index_select(2, torch.arange(8)).shape)  # 取前8行
# torch.Size([4, 3, 8, 28])
print("\n")

# ...表示取所有维度，会根据前后提供的维度推测其应该代表的维度
print(a[...].shape)  # torch.Size([4, 3, 28, 28])
print(a[0, ...].shape)  # == a[0, :, :, :]  torch.Size([3, 28, 28])
print(a[:, 1, ...].shape)  # torch.Size([4, 28, 28])
print(a[:, 1:, ..., :2].shape)  # torch.Size([4, 2, 28, 2])
print("\n")


# 利用mask来索引
x = torch.randn(3, 4)
print(x)
# tensor([[ 0.3205, -0.4509, -0.0899,  0.7717],
#         [-0.1167,  2.2696, -0.5013,  0.0376],
#         [-0.6121,  0.8619, -1.8903, -2.0185]])
mask = x.ge(0.5)  # ge == greater and equal >=
print(mask)
# tensor([[False, False, False,  True],
#         [False,  True, False, False],
#         [False,  True, False, False]])
print(torch.masked_select(x, mask))  # tensor([0.7717, 2.2696, 0.8619])
print(torch.masked_select(x, mask).shape)  # 压平为一维list，列举元素  torch.Size([3])
print("\n")

# 利用压平的index来索引，先把所有元素压平为一个list然后找到索引
src = torch.tensor([[4, 3, 5], [6, 7, 8]])
print(torch.take(src, torch.tensor([0, 2, 5])))  # tensor([4, 5, 8])












