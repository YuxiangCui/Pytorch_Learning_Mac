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
    # batch_first == True 则b放在前面
    
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
hidden_size = 16
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
            batch_first=True  # [batch, sequence_len, feature]
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [1, seq, h] => [seq, h]
        out = out.view(-1, hidden_size)
        out = self.linear(out)  # [seq, h] => [seq, 1]
        out = out.unsqueeze(dim=0)  # [seq, 1] => [1, seq, 1]
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
input = x[:, 0, :]  # x:[1, seq, 1] => input:[seq, 1]
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

#
# Epoch: 0
# Iteration: 0
# loss
# 0.5250487923622131
# Epoch: 0
# Iteration: 1000
# loss
# 0.00030210899421945214
# Epoch: 0
# Iteration: 2000
# loss
# 0.00011696002911776304
# Epoch: 0
# Iteration: 3000
# loss
# 0.0002972964139189571
# Epoch: 0
# Iteration: 4000
# loss
# 0.00025500860647298396
# Epoch: 0
# Iteration: 5000
# loss
# 0.0003937582077924162
# Epoch: 1
# Iteration: 0
# loss
# 0.0001097952263080515
# Epoch: 1
# Iteration: 1000
# loss
# 0.0002508340694475919
# Epoch: 1
# Iteration: 2000
# loss
# 0.000465849123429507
# Epoch: 1
# Iteration: 3000
# loss
# 4.268822885933332e-05
# Epoch: 1
# Iteration: 4000
# loss
# 0.0001826216612244025
# Epoch: 1
# Iteration: 5000
# loss
# 6.543553899973631e-05
# Epoch: 2
# Iteration: 0
# loss
# 0.00016076472820714116
# Epoch: 2
# Iteration: 1000
# loss
# 0.00022247030574362725
# Epoch: 2
# Iteration: 2000
# loss
# 9.835986566031352e-05
# Epoch: 2
# Iteration: 3000
# loss
# 0.00021157080482225865
# Epoch: 2
# Iteration: 4000
# loss
# 0.00024905489408411086
# Epoch: 2
# Iteration: 5000
# loss
# 0.0001475806930102408
# Epoch: 3
# Iteration: 0
# loss
# 3.677964195958339e-05
# Epoch: 3
# Iteration: 1000
# loss
# 2.1035390091128647e-05
# Epoch: 3
# Iteration: 2000
# loss
# 0.00010279594425810501
# Epoch: 3
# Iteration: 3000
# loss
# 7.863887003622949e-05
# Epoch: 3
# Iteration: 4000
# loss
# 0.00010938215564237908
# Epoch: 3
# Iteration: 5000
# loss
# 0.0001120261731557548
# Epoch: 4
# Iteration: 0
# loss
# 6.47316119284369e-05
# Epoch: 4
# Iteration: 1000
# loss
# 0.0002388201537542045
# Epoch: 4
# Iteration: 2000
# loss
# 3.4638876968529075e-05
# Epoch: 4
# Iteration: 3000
# loss
# 0.00024555623531341553
# Epoch: 4
# Iteration: 4000
# loss
# 0.00010987784480676055
# Epoch: 4
# Iteration: 5000
# loss
# 7.357023423537612e-05
# Epoch: 5
# Iteration: 0
# loss
# 0.00011029669258277863
# Epoch: 5
# Iteration: 1000
# loss
# 3.2641328289173543e-05
# Epoch: 5
# Iteration: 2000
# loss
# 2.2983511371421628e-05
# Epoch: 5
# Iteration: 3000
# loss
# 0.00028395510162226856
# Epoch: 5
# Iteration: 4000
# loss
# 0.00011884369450854138
# Epoch: 5
# Iteration: 5000
# loss
# 2.575221697043162e-05
# Epoch: 6
# Iteration: 0
# loss
# 0.00017538381507620215
# Epoch: 6
# Iteration: 1000
# loss
# 0.00017996215319726616
# Epoch: 6
# Iteration: 2000
# loss
# 0.0001108554788515903
# Epoch: 6
# Iteration: 3000
# loss
# 5.37190462637227e-05
# Epoch: 6
# Iteration: 4000
# loss
# 2.6625823011272587e-05
# Epoch: 6
# Iteration: 5000
# loss
# 3.583201760193333e-05
# Epoch: 7
# Iteration: 0
# loss
# 2.9527698643505573e-05
# Epoch: 7
# Iteration: 1000
# loss
# 4.80166963825468e-05
# Epoch: 7
# Iteration: 2000
# loss
# 0.00011473565245978534
# Epoch: 7
# Iteration: 3000
# loss
# 0.0002904343418776989
# Epoch: 7
# Iteration: 4000
# loss
# 0.00016637514636386186
# Epoch: 7
# Iteration: 5000
# loss
# 8.631061064079404e-05
# Epoch: 8
# Iteration: 0
# loss
# 0.00011888313019881025
# Epoch: 8
# Iteration: 1000
# loss
# 0.0002973313967231661
# Epoch: 8
# Iteration: 2000
# loss
# 0.0002583012683317065
# Epoch: 8
# Iteration: 3000
# loss
# 1.2457551747502293e-05
# Epoch: 8
# Iteration: 4000
# loss
# 0.0001323262695223093
# Epoch: 8
# Iteration: 5000
# loss
# 0.00011640867160167545
# Epoch: 9
# Iteration: 0
# loss
# 0.0001521813974250108
# Epoch: 9
# Iteration: 1000
# loss
# 0.00022633158368989825
# Epoch: 9
# Iteration: 2000
# loss
# 0.00028293131617829204
# Epoch: 9
# Iteration: 3000
# loss
# 0.00010200227552559227
# Epoch: 9
# Iteration: 4000
# loss
# 0.00012391981726977974
# Epoch: 9
# Iteration: 5000
# loss
# 0.00010615322389639914
