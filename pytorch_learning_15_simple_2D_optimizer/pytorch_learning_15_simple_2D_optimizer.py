# pytorch_learning_15_simple_2D_optimizer
# Himmelblau Function 用于检测优化器寻找全局最小点的能力
# f(x,y) = (x^2 + y - 11)^2 + (x + y^2 -7)^2
# f(3.0, 2.0) = 0.0
# f(-2.805118, 3.131312) = 0.0
# f(-3.779310, -3.283186) = 0.0
# f(3.584428, -1.848126) = 0.0

import torch
import numpy as np
from matplotlib import pyplot as plt


def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x y range: ', x.shape, y.shape)

X, Y = np.meshgrid(x, y)
print('X Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


