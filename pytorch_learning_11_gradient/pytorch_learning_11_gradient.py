# pytorch_learning_11_gradient
# loss and activation function
import torch
from torch.nn import functional as F

a = torch.linspace(-100,100,10)
print(a)
# tensor([-100.0000,  -77.7778,  -55.5556,  -33.3333,  -11.1111,   11.1111,
#           33.3333,   55.5555,   77.7778,  100.0000])


# Sigmoid
print(torch.sigmoid(a))
# tensor([0.0000e+00, 1.6655e-34, 7.4564e-25, 3.3382e-15, 1.4945e-05, 9.9999e-01,
#         1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00])
print("--------------------------------")

a = torch.linspace(-1, 1, 10)
print(a)
# tensor([-1.0000, -0.7778, -0.5556, -0.3333, -0.1111,  0.1111,  0.3333,  0.5556,
#          0.7778,  1.0000])


# Tanh
print(torch.tanh(a))
# tensor([-0.7616, -0.6514, -0.5047, -0.3215, -0.1107,  0.1107,  0.3215,  0.5047,
#          0.6514,  0.7616])
print("--------------------------------")


# ReLU
print(torch.relu(a))
# tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1111, 0.3333, 0.5556, 0.7778,
#         1.0000])

print("==================================")

# Loss
# Mean Squared Error(MSE) & Cross Entropy Loss

# Mean Squared Error(MSE)
x = torch.ones(1)
w = torch.full([1], 2)
mse = F.mse_loss(x*w, torch.ones(1))  # pred,label
print(mse)  # tensor(1.)
# print(torch.autograd.grad(mse, [w]))  # 此时w变量（所有自变量）不具备求导特性，梯度

print(w.requires_grad_())  # tensor([2.], requires_grad=True)
# 或者w = torch.full([1], 2)声明时直接加上require_grad_属性
# print(torch.autograd.grad(mse, [w]))  # 此时w更新了，但是动态图没变

# 求解梯度法一
# torch.autograd.grad(loss, [w1,w2,w3,...])
# 直接返回各个变量的梯度信息
mse = F.mse_loss(x*w, torch.ones(1))  # pred,label 重新构建动态图
print(torch.autograd.grad(mse, [w]))
# (tensor([2.]),)

# 求解梯度法二
# loss.backward()
# 梯度信息保存在动态图各个变量中，手动进行查看
mse = F.mse_loss(x*w, torch.ones(1))
mse.backward()
print(w.grad)
# tensor([2.])
print("--------------------------------")
# Cross Entropy Loss 分类问题
# binary 二分类
# multi-class 多分类
# + soft-max 一般搭配使用
# Logistic Regression 章节详讲

# soft-max: soft version of max
# soft-max' = d_p_i / d_a_j 分别对于各个类别求偏导
# soft-max'(i) = p_i(1-p_j)  i==j  "+"
# soft-max'(i) = -p_j*p_i          "-"
# 保证各个概率和为1

a = torch.rand(3)
a.requires_grad_()
print(a)
# tensor([0.3573, 0.6056, 0.3825], requires_grad=True)
print(a.shape)

p = F.softmax(a, dim=0)
# p.backward() 这里直接使用，前面的动态图已经被自动清除，须要在前一次backward中加参数retain_graph = True

p = F.softmax(a, dim=0)
print(torch.autograd.grad(p[1], [a], retain_graph=True))
# (tensor([-0.1172,  0.2374, -0.1202]),)
print(torch.autograd.grad(p[2], [a]))
# (tensor([-0.0938, -0.1202,  0.2139]),)
# 对应前面的正负关系

# 法二
p = F.softmax(a, dim=0)
p[1].backward()
print(a.grad)