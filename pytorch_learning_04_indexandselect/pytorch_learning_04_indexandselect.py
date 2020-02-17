import torch
a = torch.rand(4, 3, 28, 28)

# 指定通道取出
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 0, 2, 4])


# 通用形式  start:end:step

# 连续取
# 一个:，表示取全部
# :加数字表示从0取到数字之前，:2表示0，1，不包含最后一个       选取部分所在的方向为 ==>
# 数字加:表示从数字取到最后，rgb，1:，表示gb，包含起始位       选取部分所在的方向为 <==
# a:b表示取区间内部，[a, b)，左开右闭
# 负数表示反向索引    [0, 1, 2]   [-3, -2, -1]
print(a[:2, :2].shape)
print(a[:2, :2, :, :].shape)
print(a[:2, 1:, :, :].shape)
print(a[:2, 0:2, :, :].shape)
print(a[:2, -1:, :, :].shape)


# 间隔取
print(a[:, :, 0:28:2, 0:28:2].shape)
print(a[:, :, ::2, ::2].shape)

# 选取指定的位置
print(a.index_select(0, torch.tensor([0, 2, 3])).shape)  # 这里的参数为指定的位置，第023张图片
print(a.index_select(1, torch.tensor([1, 2])).shape)  # 这里的参数为指定的位置，GB通道
print(a.index_select(2, torch.arange(28)).shape)  # 取前28行
print(a.index_select(2, torch.arange(8)).shape)  # 取前8行
# ...表示取所有维度，会根据前后提供的维度推测其应该代表的维度
print(a[...].shape)
print(a[0, ...].shape)  # == a[0, :, :, :]
print(a[:, 1, ...].shape)
print(a[:, 1:, ..., :2].shape)


# 利用mask来索引
x = torch.randn(3, 4)
print(x)
mask = x.ge(0.5)  # ge == greater and equal >=
print(mask)
print(torch.masked_select(x, mask))
print(torch.masked_select(x, mask).shape)  # 压平为一维list，列举元素

# 利用压平的index来索引，先把所有元素压平为一个list然后找到索引
src = torch.tensor([[4, 3, 5], [6, 7, 8]])
print(torch.take(src, torch.tensor([0, 2, 5])))












