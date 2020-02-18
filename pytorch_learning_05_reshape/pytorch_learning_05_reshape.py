import torch

a = torch.rand(4, 1, 28, 28)
print(a.shape)  # torch.Size([4, 1, 28, 28])


# view / reshape
# view前后元素数不变，a.numel()，size要相同
print(a.view(4, 1*28*28))
print(a.view(4, 1*28*28).shape)  # 每张图片展开 torch.Size([4, 784])
print(a.view(4*28, 28))  # 只关注每一行数据
print(a.view(4*1, 28, 28))  # 只关注每一张feature map而不关注来自哪一个通道

# view 展开丢失原来的维度信息，再次强行恢复会导致数据混乱，要记住维度顺序
# (4,1,28,28) => view(4,28*28) => view(4,28,28,1)


# squeeze / un-squeeze
# un-squeeze 增加新的维度，[0,1,2][-3,-2,-1]
# 取值范围为[-a.dim()-1,a.dim()+1)
# e.g. [4, 1, 28, 28]  [-5,5) => -5,-4,-3,-2,-1,0,1,2,3,4
#     [0, 1, 2,  3, 4] 非负数，在原来对应位前增加新的维度
#    [-5,-4,-3, -2,-1] 负数，在原来对应位之后增加新的维度
print(a.unsqueeze(0).shape)  # torch.Size([1, 4, 1, 28, 28])
print(a.unsqueeze(2).shape)  # torch.Size([4, 1, 1, 28, 28])
print(a.unsqueeze(4).shape)  # torch.Size([4, 1, 28, 28, 1])
print(a.unsqueeze(-1).shape)  # torch.Size([4, 1, 28, 28, 1])
print(a.unsqueeze(-3).shape)  # torch.Size([4, 1, 1, 28, 28])
print(a.unsqueeze(-5).shape)  # torch.Size([1, 4, 1, 28, 28])



# transpose / t / permute


# expand / repeat






