# pytorch_learning_25_LSTM_functions

#     rnn = nn.LSTM(100,10)
#     __init__
#     # input size(embedding 的维度，表达向量的维度), hidden size(隐藏层维度,C和h维度相同), num layers LSTM层数，默认是1
#
#     out, (ht, ct) = lstm(x, (h0, c0))
#     # x: [seq_len, b, word_vec] 一次性给定所有输入 e.g. [5, 3, 100] 三句话，每句话五个单词长度，每个单词用一个100维向量表示
#     # h0 / ht: [num layers, b, h dim] h0可以不提供，默认全零开始， ht为最后的返回隐藏层状态
#     # out: [seq_len, b, h dim] 所有的h


import torch

