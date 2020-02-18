# add/minus/multiply/divide
# matmul
# pow
# sqrt/rsqrt
# round

import torch

a = torch.rand(3, 4)
b = torch.rand(4)

# + - * /
# broadcast机制会自动进行调整
# 可以直接用重载的运算符号来进行运算，替代指令
# print(a+b)
# print(torch.add(a,b))
print(torch.all(torch.eq(a+b, torch.add(a,b))))
# print(a-b)
# print(torch.sub(a,b))
print(torch.all(torch.eq(a-b, torch.sub(a,b))))
# 这里是element-wise乘法，对应元素相乘
# print(a*b)
# print(torch.mul(a,b))
print(torch.all(torch.eq(a*b, torch.mul(a,b))))
# print(a/b)
# print(torch.div(a,b))
print(torch.all(torch.eq(a/b, torch.div(a,b))))


# matmul
# 这里是常用的矩阵乘法
a = 3 * torch.ones(2, 2)
b = torch.ones(2, 2)

c = 3 * torch.ones(2, 2, 2)
d = torch.ones(2, 2, 2)

print(a*b)
# tensor([[3., 3.],
#         [3., 3.]])
print(torch.mm(a, b))  # 这种表达方式只适用于二维
# tensor([[6., 6.],
#         [6., 6.]])
# print(torch.mm(c, d))
print(torch.matmul(a, b))
# tensor([[6., 6.],
#         [6., 6.]])
print(a@b)
# tensor([[6., 6.],
#         [6., 6.]])

a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
# matmul其实代表支持多个矩阵相乘并行，所以仍然是最后两维的矩阵相乘
print(torch.matmul(a, b).shape)  # torch.Size([4, 3, 28, 32])

c = torch.rand(4, 1, 64, 32)
d = torch.rand(4, 2, 64, 32)
print(torch.matmul(a, c).shape)  # torch.Size([4, 3, 28, 32])
# print(torch.matmul(a, d).shape)  # 不满足broadcast


# Power

a = 3 * torch.ones(2, 2)
print(a.pow(2))
print(a**2)  # **表示幂
# tensor([[9., 9.],
#         [9., 9.]])

aa = a**2
print(aa.sqrt())
# tensor([[3., 3.],
#         [3., 3.]])
print(aa.rsqrt())
# tensor([[0.3333, 0.3333],
#         [0.3333, 0.3333]])  平方根的倒数
print(aa**0.5)
# tensor([[3., 3.],
#         [3., 3.]])

# exp log

a = torch.exp(torch.ones(2, 2))
print(a)
# tensor([[2.7183, 2.7183],
#         [2.7183, 2.7183]])
# log默认为自然底数，log2，log10
print(torch.log(a))
# tensor([[1.0000, 1.0000],
#         [1.0000, 1.0000]])


# approximation
# floor() / ceil() 向下向上取整
a = torch.tensor(3.14)
print(a.floor())
print(a.ceil())

# round() 四舍五入
a = torch.tensor(3.499)
b = torch.tensor(3.501)
print(a.round())
print(b.round())

# trunc() / frac() 取整数部分 取小数部分
a = torch.tensor(3.14)
print(a.trunc())
print(a.frac())


# clamp 裁剪
# gradient clipping 梯度不稳定时常用

grad = torch.rand(2, 3) * 15
print(grad)
# tensor([[13.9204,  1.7256,  9.6349],
#         [14.2555,  6.4697, 12.2107]])
print(grad.max())
# tensor(14.2555)
print(grad.median())
# tensor(9.6349)
print(grad.clamp(5))
# tensor([[13.9204,  5.0000,  9.6349],
#         [14.2555,  6.4697, 12.2107]])
print(grad.clamp(10))
# tensor([[13.9204, 10.0000, 10.0000],
#         [14.2555, 10.0000, 12.2107]])
print(grad.clamp(2,10))
# tensor([[10.0000,  2.0000,  9.6349],
#         [10.0000,  6.4697, 10.0000]])


