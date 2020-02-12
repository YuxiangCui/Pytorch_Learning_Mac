import torch
import numpy as np
import csv

# sFileName = 'Linear_data.csv'
#
# with open(sFileName, newline='', encoding='UTF-8') as csv_file:
#     rows = csv.reader(csv_file)
#     for row in rows:
#         print(','.join(row))


# y = wx + b
# Loss
def compute_error_for_line_given_points(b, w, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))


# Gradient
def step_grad(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # 偏导数
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))

    # 更新
    new_b = b_current - (learning_rate * b_gradient)
    new_w = w_current - (learning_rate * w_gradient)
    return [new_b, new_w]


# 迭代优化器
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iter):
    b = starting_b
    w = starting_w
    for i in range(num_iter):
        b, w = step_grad(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    # 读取第一行有些问题，出现nan，暂时没有找到原因，选择跳过第一行
    points = np.genfromtxt("Linear_data.csv", delimiter=',', skip_header=1)
    # print(points[0][0])
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0},w = {0},error = {2}"
          .format(initial_b, initial_w, points)
          )
    print("Running....")

    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iteration b = {1}, w = {2},error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points)
                 )
          )


if __name__ == '__main__':
    print(torch.__version__)
    run()
