# pytorch_learning_17_simple_multi_classification

import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms as transforms

batch_size = 200
learning_rate = 0.01
epochs = 10

w_1, b_1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
w_2, b_2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
w_3, b_3 = torch.randn(10, 200, requires_grad=True), torch.zeros(10, requires_grad=True)

torch.nn.init.kaiming_normal_(w_1)
torch.nn.init.kaiming_normal_(w_2)
torch.nn.init.kaiming_normal_(w_3)  # 初始化不好的话可能会导致梯度离散等问题


def forward(x):
    x = x@w_1.t() + b_1
    x = F.relu(x)
    x = x @ w_2.t() + b_2
    x = F.relu(x)
    x = x @ w_3.t() + b_3
    x = F.relu(x)  # 这里有没有无所谓，但是不可以是sigmoid或者soft-max，因为后面的函数中有包含
    return x


# Load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        '../pytorch_learning_02_MNIST/mnist_data',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        '../pytorch_learning_02_MNIST/mnist_data/',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size,
    shuffle=False)

optimizer = torch.optim.SGD([w_1, b_1, w_2, b_2, w_3, b_3], lr=learning_rate)
criteon = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)

        logits = forward(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch:{}[{}/{} ({:.0f}%]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        test_loss += criteon(logits, target).item()
        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))



#
# Train Epoch:0[0/60000 (0%]      Loss:2.599304
# Train Epoch:0[20000/60000 (33%] Loss:1.317541
# Train Epoch:0[40000/60000 (67%] Loss:0.889058
#
# Test Set: Average loss: 0.0040, Accuracy: 7498/10000 (75%)
# Train Epoch:1[0/60000 (0%]      Loss:0.648577
# Train Epoch:1[20000/60000 (33%] Loss:0.926337
# Train Epoch:1[40000/60000 (67%] Loss:0.772714
#
# Test Set: Average loss: 0.0036, Accuracy: 7993/10000 (80%)
# Train Epoch:2[0/60000 (0%]      Loss:0.617835
# Train Epoch:2[20000/60000 (33%] Loss:0.716179
# Train Epoch:2[40000/60000 (67%] Loss:0.581642
#
# Test Set: Average loss: 0.0026, Accuracy: 8317/10000 (83%)
# Train Epoch:3[0/60000 (0%]      Loss:0.542941
# Train Epoch:3[20000/60000 (33%] Loss:0.491907
# Train Epoch:3[40000/60000 (67%] Loss:0.604417
#
# Test Set: Average loss: 0.0023, Accuracy: 8407/10000 (84%)
# Train Epoch:4[0/60000 (0%]      Loss:0.538913
# Train Epoch:4[20000/60000 (33%] Loss:0.426344
# Train Epoch:4[40000/60000 (67%] Loss:0.485947
#
# Test Set: Average loss: 0.0022, Accuracy: 8468/10000 (85%)
# Train Epoch:5[0/60000 (0%]      Loss:0.470551
# Train Epoch:5[20000/60000 (33%] Loss:0.493053
# Train Epoch:5[40000/60000 (67%] Loss:0.487335
#
# Test Set: Average loss: 0.0021, Accuracy: 8516/10000 (85%)
# Train Epoch:6[0/60000 (0%]      Loss:0.405175
# Train Epoch:6[20000/60000 (33%] Loss:0.493235
# Train Epoch:6[40000/60000 (67%] Loss:0.451716
#
# Test Set: Average loss: 0.0021, Accuracy: 8533/10000 (85%)
# Train Epoch:7[0/60000 (0%]      Loss:0.350127
# Train Epoch:7[20000/60000 (33%] Loss:0.423546
# Train Epoch:7[40000/60000 (67%] Loss:0.491799
#
# Test Set: Average loss: 0.0020, Accuracy: 8566/10000 (86%)
# Train Epoch:8[0/60000 (0%]      Loss:0.274815
# Train Epoch:8[20000/60000 (33%] Loss:0.398818
# Train Epoch:8[40000/60000 (67%] Loss:0.325850
#
# Test Set: Average loss: 0.0020, Accuracy: 8584/10000 (86%)
# Train Epoch:9[0/60000 (0%]      Loss:0.368150
# Train Epoch:9[20000/60000 (33%] Loss:0.292812
# Train Epoch:9[40000/60000 (67%] Loss:0.256455
#
# Test Set: Average loss: 0.0019, Accuracy: 8603/10000 (86%)
#
# Process finished with exit code 0

