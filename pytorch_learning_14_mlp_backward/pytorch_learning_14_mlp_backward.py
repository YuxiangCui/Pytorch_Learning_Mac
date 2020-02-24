# pytorch_learning_14_mlp_backward

# import torch
# 多层感知机的反向传播
# 通过当前层的输出以及之前网络的参数信息，可以得到最后的error与当前层参数的偏导信息，从而进行调整

# for the output layer node k
# partial_E / partial_W_ij = O_j * delta_k
# where delta_k = O_k * (1 - O_k) * (O_k - t_k)

# for the hidden layer up node j
# partial_E / partial_W_ij = O_i * delta_j
# where delta_j = O_j * (1 - O_j) * SUM_k(delta_k * W_jk)



