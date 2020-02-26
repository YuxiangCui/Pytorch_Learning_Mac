# pytorch_learning_21_nn_module

# nn.module
# Linear, ReLU, Sigmoid...模块
# nn.Sequential 网络容器
# 更好的网络参数管理 list(net.parameters())  list(net.named_parameters())  或者替换list为dict
# 调用参数也方便 optimizer = optim.SGD(net.parameters(), lr=1e-3)
# children 直接连接的子节点，modules 所有的节点
# to(device)命令  注意区别模块和tensor的使用
# save & load
# train & test
# 自定义的话也要继承自 nn.module 例如nn里面没有的flatten操作层，reshape操作等
# nn.Parameter()包装自己的变量进入优化器参数内，require_grad
import torch
