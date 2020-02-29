import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt

'''
    rnn = nn.RNN(100,10)  
    # input size(embedding 的维度，表达向量的维度), hidden size(隐藏层维度), num layers RNN层数，默认是1
    
    out, ht = forward(x, h0)
    # x: [seq_len, b, word_vec] 一次性给定所有输入 e.g. [5, 3, 100] 三句话，每句话五个单词长度，每个单词用一个100维向量表示
    # h0 / ht: [num layers, b, h dim] h0可以不提供，默认全零开始， ht为最后的返回隐藏层状态
    # out: [seq_len, b, h dim] 所有的h
    
    rnn._parameters.keys()
    print(rnn._parameters.keys())

    # 单层RNN
    rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
    print(rnn)
    x = torch.randn(10,3,100)  # 单词个数，batch_size，单词向量维度
    out, h = rnn(x,torch.zeros(1,3,20))
    print(out.shape,h.shape)
    # torch.Size([10, 3, 20]) torch.Size([1, 3, 20])
    
    --------------------------------------------------------------------------------
    # RNN层数加深的话
    # h[0]增大，对应层数增大，但是out不变，只返回最后一层RNN的h
    # out为时间序列方向，横向，所有，ht为最后时刻所有的RNN隐藏层参数，纵向，所有
    
    rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
    print(rnn)
    x = torch.randn(10,3,100)  # 单词个数，batch_size，单词向量维度
    out, h = rnn(x,torch.zeros(4,3,20))
    print(out.shape,h.shape)
    # torch.Size([10, 3, 20]) torch.Size([4, 3, 20])
    
    --------------------------------------------------------------------------------

    nn.RNNCell
    # 逐个时间戳数据输入
    # input size(embedding 的维度，表达向量的维度), hidden size(隐藏层维度), num layers RNN层数，默认是1
    ht = rnncell(xt, ht_1)
    xt: [batch_size, word_vec]
    ht_1 / ht: [num_layers, batch_size, h_dim]
    out: torch.stack([h1, h2, ..., ht])
    
    cell1 = nn.RNNCell(100, 20)
    h1 = torch.zeros(3, 20)
    for xt in x:
        h1 = cell1(xt, h1)
    print(h1.shape)
    # torch.Size([3,20])
    
'''
num_time_steps = 50
input_size = 1
hidden_size = 10
output_size = 1
lr = 0.01
epochs = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, h = self.rnn(x, hidden_prev)

        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = torch.zeros(1, 1, hidden_size)
for epoch in range(epochs):
    for iter in range(6000):
        start = np.random.randint(3, size=1)[0]
        time_steps = np.linspace(start, start + 10, num_time_steps)
        data = np.sin(time_steps)
        data = data.reshape(num_time_steps, 1)
        x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
        y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

        output, hidden_prev = model(x, hidden_prev)
        hidden_prev = hidden_prev.detach()

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 1000 == 0:
            print(" Epoch: {} Iteration: {} loss {}".format(epoch, iter, loss.item()))

    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

predictions = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    (pred, hidden_prev) = model(input, hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], predictions)
plt.show()