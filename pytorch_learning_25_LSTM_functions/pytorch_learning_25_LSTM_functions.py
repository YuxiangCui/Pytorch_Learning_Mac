# pytorch_learning_25_LSTM_functions

#     rnn = nn.LSTM(100,10)
#     __init__
#     # input size(embedding 的维度，表达向量的维度), hidden size(隐藏层维度,C和h维度相同), num layers LSTM层数，默认是1
#
#     out, (ht, ct) = lstm(x, (h0, c0))
#     # x: [seq_len, b, word_vec] 一次性给定所有输入 e.g. [5, 3, 100] 三句话，每句话五个单词长度，每个单词用一个100维向量表示
#     # h0 / ht: [num layers, b, h dim] h0可以不提供，默认全零开始， ht为最后的返回隐藏层状态
#     # out: [seq_len, b, h dim] 所有的h

#     nn.LSTMCell
#     # 逐个时间戳数据输入
#     __init__
#     # input size(embedding 的维度，表达向量的维度), hidden size(隐藏层维度), num layers RNN层数，默认是1
#     ht,ct = lstmcell(xt, [ht_1, ct_1])
#     xt: [batch_size, word_vec]
#     ht_1 / ht / ct_1 / ct: [num_layers, batch_size, h_dim]
#     out: torch.stack([h1, h2, ..., ht])

import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
print(lstm)  # LSTM(100, 20, num_layers=4)
x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)
print(out.shape, h.shape, c.shape)
# torch.Size([10, 3, 20]) torch.Size([4, 3, 20]) torch.Size([4, 3, 20])

print('one layer lstm')
cell = nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)

for xt in x:
    h, c = cell(xt, [h, c])

print(h.shape, c.shape)
# torch.Size([3, 20]) torch.Size([3, 20])

print('two layer lstm')
cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)
for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(h1, [h2, c2])

print(h2.shape, c2.shape)
# torch.Size([3, 20]) torch.Size([3, 20])


