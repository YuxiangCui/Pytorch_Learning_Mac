# view / reshape
# squeeze / un-squeeze
# expand / repeat
# transpose / t / permute
import torch

a = torch.rand(4, 1, 28, 28)
print(a.shape)  # torch.Size([4, 1, 28, 28])


# view / reshape
# view前后元素数不变，a.numel()，size要相同
print(a.view(4, 1*28*28))
print(a.view(4, 1*28*28).shape)  # 每张图片展开 torch.Size([4, 784])
print(a.view(4*28, 28))  # 只关注每一行数据
print(a.view(4*1, 28, 28))  # 只关注每一张feature map而不关注来自哪一个通道

# view 展开丢失原来的维度信息，再次强行恢复会导致数据混乱
# 要记住维度顺序，保证view前后维度顺序相同
# (4,1,28,28) => view(4,28*28) => view(4,28,28,1)


# squeeze / un-squeeze

# un-squeeze 增加新的维度，[0,1,2][-3,-2,-1]
# 取值范围为[-a.dim()-1,a.dim()+1)
# e.g. [4, 1, 28, 28]  [-5,5) => -5,-4,-3,-2,-1,0,1,2,3,4，左闭右开
#     [0, 1, 2,  3, 4] 非负数，在原来对应位前增加新的维度
#    [-5,-4,-3, -2,-1] 负数，在原来对应位之后增加新的维度
print(a.unsqueeze(0).shape)  # torch.Size([1, 4, 1, 28, 28])
print(a.unsqueeze(2).shape)  # torch.Size([4, 1, 1, 28, 28])
print(a.unsqueeze(4).shape)  # torch.Size([4, 1, 28, 28, 1])
print(a.unsqueeze(-1).shape)  # torch.Size([4, 1, 28, 28, 1])
print(a.unsqueeze(-3).shape)  # torch.Size([4, 1, 1, 28, 28])
print(a.unsqueeze(-5).shape)  # torch.Size([1, 4, 1, 28, 28])

# e.g. (1)
example = torch.tensor([1, 2])  # [2]
print(example.unsqueeze(-1))
# [2,1]
# tensor([[1],
#         [2]])
print(example.unsqueeze(0))
# [1,2]
# tensor([[1, 2]])

# e.g.(2)
f = torch.rand(4, 32, 14, 14)
b = torch.rand(32)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)  # [32]=>[32,1]=>[32,1,1]=>[1,32,1,1]
print(b.shape) # [1,32,1,1] 增加扩张操作便可以形成与f同样规格的向量使之可以相加，一般用作偏置等操作

# squeeze 删减维度，按照指定的参数去删减/挤压，不给参数则能删多少删多少，给的维度不能删减/挤压则不变

print(b.squeeze().shape)  # torch.Size([32]) 能挤压多少挤压多少
print(b.squeeze(0).shape)  # torch.Size([32, 1, 1])
print(b.squeeze(1).shape)  # torch.Size([1, 32, 1, 1]) 这里不能删减该维度
print(b.squeeze(-4).shape)  # torch.Size([32, 1, 1])



# expand / repeat

# expand : broadcasting 改变形式，返回视图（view），不复制数据，得到的扩张结果在内存中不连续
# repeat : memory copied 复制数据，增大数据量，得到的扩张结果在数据中连续

a = torch.rand(4, 32, 14, 14)
b = torch.rand(32)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)  # [32]=>[32,1]=>[32,1,1]=>[1,32,1,1]

# expand
# expand前后维度要相同，对于须要扩张的维度，要保证 1->N，不可以n->N
print(b.expand(4, 32, 14, 14).shape)  # torch.Size([4, 32, 14, 14])
# -1表示该维度不变
print(b.expand(-1, -1, -1, -1).shape)  # torch.Size([1, 32, 1, 1])
# 给定负数会出BUG，没有实际意义
print(b.expand(-1, 32, -1, -4).shape)  # torch.Size([1, 32, 1, -4])
# 不可以n->N
# print(b.expand(-1, 33, -1, -1).shape)  # torch.Size([1, 32, 1, 1])


# repeat
# repeat的参数表示的是该维度copy的次数
print(b.repeat(4, 32, 1, 1).shape)  # torch.Size([4, 1024, 1, 1])
print(b.repeat(4, 1, 14, 14).shape)  # torch.Size([4, 32, 14, 14])


# transpose / t / permute

# t 转置 只适用于二维矩阵
a = torch.randn(3, 4)
print(a.shape)  # torch.Size([3, 4])
print(a.t().shape)  # torch.Size([4, 3])

# transpose 交换指定两个维度
a = torch.randn(4, 3, 32, 32)
# a.transpose(1, 3)  [4, 3, 32, 32] => [4, 32, 32, 3]
# 交换维度后，数据存储不连续，且view丢弃了原来的维度顺序信息，重新强制分开会出错
# a_0 = a.transpose(1, 3).view(4, 3*32*32).view(4, 3, 32, 32)

a_1 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 3, 32, 32)
a_2 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 32, 32, 3).transpose(1, 3)

print(a_1.shape)
print(a_2.shape)
print(torch.all(torch.eq(a, a_1)))
print(torch.all(torch.eq(a, a_2)))


# permute
a = torch.randn(4, 3, 28, 32)
# 例如图片处理中要保证H,W的顺序，保证图片的格式(numpy的图片存储格式也是这样的)，但是利用transpose进行变换，需要两步处理
# [B, C, H, W] => [B, W, H, C] => [B, H, W, C]
a_1 = a.transpose(1, 3).transpose(1, 2)
a_2 = a.permute(0, 2, 3, 1)
print(a_1.shape)
print(a_2.shape)
# 会出现内存打乱，可能需要contiguous()
