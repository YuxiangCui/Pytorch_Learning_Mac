# pytorch_learning_16_cross_entropy
#
# entropy：表达不确定性
# entropy = -SUM{P(i) * log(P(i))}
# uncertainty ++, entropy --
# 来自于哪个变量的不确定性越大，即各个变量的概率越均衡，熵越大
# 概率存在明显的偏移，则蕴含的信息越多，熵越低
import torch
from torch.nn import functional as F

# entropy = -SUM{P(i) * log(P(i))}
a = torch.full([4], 1/4)
b = torch.tensor([0.1, 0.1, 0.1, 0.7])
c = torch.tensor([0.001, 0.001, 0.001, 0.997])

print(-(a * torch.log2(a)).sum())  # tensor(2.)
print(-(b * torch.log2(b)).sum())  # tensor(1.3568)
print(-(c * torch.log2(c)).sum())  # tensor(0.0342)


# cross entropy = -SUM{P(i) * log(Q(i))}
# H(p,q) = H(p) + D_kl(p|q)  (D_kl(p|q) 表示两个分布的差异)
# P = Q 时，H(p,q) = H(p)
# 预测用one-hot进行编码时，H(p,q) = D_kl(p|q)  (1log1=0)

# e.g. 二分类问题
# H(p,q) = - a * logb - (1 - a) * log(1 - b) 其中 a = P (m) b = Q(m)
# 得到的熵越小，表示P和Q越接近， 越接近真值

# 交叉熵不会出现sigmoid + MSE中会出现的饱和，梯度消失等问题
# 梯度更为明显，收敛更快
# 但是当交叉熵效果不好时可以对MSE进行尝试，形式更为简单

# numerical stability 数值不稳定问题
# 一般情况下把soft-max和log运算等都包含在cross-entropy函数里面

x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x@w.t()  # [1, 10]
pred = F.softmax(logits, dim=1)  # [1, 10]
pred_log = torch.log(pred)
print(logits)
print(F.cross_entropy(logits, torch.tensor([3])))
print(F.nll_loss(pred_log, torch.tensor([3])))  # 结果相同，注意输入的区别  negative log likelihood loss


