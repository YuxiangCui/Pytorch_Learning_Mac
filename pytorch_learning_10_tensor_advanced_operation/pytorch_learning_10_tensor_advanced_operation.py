# pytorch_learning_10_tensor_advanced_operation
# where
# gather

import torch

# torch.where(condition, x, y)
# return a tensor of elements selected from either x or y, depending on condition
#          { x_i  if condition_i
#  out_i = |
#          { y_i  otherwise
# for循环嵌套在cpu上运行，换成where可以进行gpu加速，但是前提是cond生成

cond = torch.rand(2, 2)
print(cond)
# tensor([[0.4150, 0.3991],
#         [0.6582, 0.9844]])
a = torch.zeros(2, 2)
b = torch.ones(2, 2)
c = torch.where(cond > 0.5, a, b)
print(c)
# tensor([[1., 1.],
#         [0., 0.]])


# torch.gather(input, dim, index, out=None) -> Tensor
# gathers value along an axis specified by dim
# input: 查表, dim: 查表所在的维度, index:索引号
# 相对label与直接对应的分类编号或者名称不对应，需要进行查表
# relative => global 利用gpu进行查表

prod = torch.rand(4, 10)
idx = prod.topk(dim=1, k=3)
print(idx)
# torch.return_types.topk(
# values=tensor([[0.8976, 0.8762, 0.8178],
#         [0.8364, 0.7986, 0.6991],
#         [0.9836, 0.7382, 0.6904],
#         [0.9498, 0.9230, 0.8575]]),
# indices=tensor([[0, 7, 6],
#         [0, 4, 2],
#         [6, 5, 9],
#         [6, 8, 5]]))
idx = idx[1]
label = torch.arange(0, 20, 2) + 100
print(label)
# tensor([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
print(torch.gather(label.expand(4, 10), dim=1, index=idx.long()))
# tensor([[100, 114, 112],
#         [100, 108, 104],
#         [112, 110, 118],
#         [112, 116, 110]])


