# pytorch_learning_18_visdom

import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms as transforms
from visdom import Visdom

viz = Visdom()

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

global_step = 0
viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                   legend=['loss', 'acc.']))

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)

        logits = forward(data)
        loss = criteon(logits, target)

        viz.line([loss.item()], [global_step], win='train_loss', update='append')

        global_step += 1
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

    viz.line([[test_loss, correct / len(test_loader.dataset)]],
             [global_step], win='test', update='append')
    viz.images(data.view(-1, 1, 28, 28), win='x')
    viz.text(str(pred.numpy()), win='pred',
             opts=dict(title='pred'))


    test_loss /= len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
