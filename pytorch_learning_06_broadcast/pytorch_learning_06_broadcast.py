# Broadcast 自动扩展
# expand
# without copying data
# 前面插入维度，然后把该维度扩展到相同数目
# 当前维度，要满足后面大小一致，也就是说小维度一致，大维度进行扩张,且必须从1开始扩张
# 前面没有这个维度，加一个维度
# 前面有这个维度，扩张为等同数目
# 满足相加操作的需求，且不拷贝数据，节省内存
# 相当于 un-squeeze 和 expand
# a : [4, 32, 14, 14]
# b : [32] => [32, 1, 1](手动) => [1, 32, 1, 1] => [4, 32, 14 , 14]
import torch
a = torch.rand(4, 32, 14, 14)
b_1 = torch.rand(1, 32, 1, 1)  # 可
b_2 = torch.rand(14, 14)  # 可
b_3 = torch.rand(2, 32, 14, 14)  # 不可

print((a+b_1).shape)
print((a+b_2).shape)
# print((a+b_3).shape)

a = torch.rand(4, 32, 1, 14)
b_1 = torch.rand(1, 32, 14, 1)  # 可

print((a+b_1).shape)  # 两边都可以补
