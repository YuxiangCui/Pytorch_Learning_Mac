# 引入预训练模型
# 迁移学习
# 当数据量很少时，模型又相对于数据复杂度过于复杂，很容易产生过拟合
# 考虑当前数据属于更大的数据库中，例如ImageNet，那么根据预训练模型进行fine-tune可以得到更好地效果

import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import torchvision
import torchvision.transforms as transforms
import argparse
# from pytorch_learning_27_DIY_Dataset_CNN import ResNet
from torchvision.models import resnet18
from DIY_dataloader import DIY_dataset
from torch.utils.data import Dataset, DataLoader


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(1234)
# 超参数设置
EPOCH = 10
pre_epoch = 0
BATCH_SIZE = 32
LR = 1e-3
root = '../pytorch_learning_dataset/pokeman'

viz = visdom.Visdom()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1,shape)


def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total


train_db = DIY_dataset(root, 224, mode='training')
val_db = DIY_dataset(root, 224, mode='validation')
test_db = DIY_dataset(root, 224, mode='testing')

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_db, batch_size=BATCH_SIZE, num_workers=2)

# 模型定义-ResNet
# net = ResNet().to(device)
trained_model = resnet18(pretrained=True)
net = nn.Sequential(*list(trained_model.children())[:-1],  # torch.Size([2, 512, 1, 1])
                      Flatten(),
                      nn.Linear(512, 5)
                      ).to(device)
# x = torch.randn(2, 3, 224, 224)
# print(model(x).shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

# 训练
if __name__ == "__main__":

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    print("Start Training, Resnet!")  # 定义遍历数据集的次数

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        for i, data in enumerate(train_loader):
             # 准备数据
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            #
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        # 每训练2个epoch进行一次evaluate
        if epoch % 2 == 0:
            val_acc = evaluate(net, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

                torch.save(net.state_dict(), 'best.mdl')

            viz.line([val_acc], [global_step], win='val_acc', update='append')



    print('best accuracy: ', best_acc, ' best epoch: ', best_epoch)
    net.load_state_dict(torch.load('best.mdl'))
    print('loaded from check point')

    test_acc = evaluate(net, test_loader)
    print('test accuracy: ', test_acc)


# best accuracy:  0.9527896995708155  best epoch:  6
# loaded from check point
# test accuracy:  0.9316239316239316
