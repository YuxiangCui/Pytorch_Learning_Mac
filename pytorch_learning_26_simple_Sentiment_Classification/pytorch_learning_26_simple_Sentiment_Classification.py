# pytorch_learning_26_simple_Sentiment_Classification

# Google CoLab
# K80 可以使用12h 会清洗

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torchtext import data, datasets


print('GPU: ', torch.cuda.is_available())
torch.manual_seed(123)


TEXT = data.Field(tokensize='spacy')
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print('len of train data: ', len(train_data))
print('len of test data: ', len(test_data))
print(train_data.examples[15].text)
print(train_data.examples[15].label)

# word_to_vec  glove
TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

batchsz = 80
device = torch.device('cuda')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batchsz,
    device=device
)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        # [0-10001] => [100]  逐词进行编码，将单词维度降低
        self.embbeding = nn.Embedding(vocab_size, embedding_dim)
        # [100] => [256]  LSTM,这里是双向的
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        # [256*2] => [1]  全连接得到最后的判断结果
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # [seq, b, 1] => [seq, b, 100]  逐词进行编码
        embedding = self.dropout(self.embbeding(x))
        # output: [seq, b, hid_dim*2]
        # hidden / h: [num_layers*2, b, hid_dim]
        # cell / c: [num_layers*2, b, hid_dim]
        output, (hidden, cell) = self.rnn(embedding)
        # [num_layers*2, b, hid_dim] => 2 * [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat(hidden[-2], hidden[-1], dim=1)
        # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out


rnn = RNN(len(TEXT.vocab), 100, 256)

pretrained_embedding = TEXT.vocab.vectors
print('pretrained_embedding: ', pretrained_embedding.shape)
rnn.embbeding.weight.data.copy_(pretrained_embedding)
print('embedding layer inited')

def train(rnn, iterator, optimizer, criterion):
    avg_acc = []
    rnn.train()

    for i, batch in enumerate(iterator):
        # [seq, b] => [b, 1] => [b]
        pred = rnn(batch.text).squeeze(1)
        loss = criterion(pred, batch.label)
        acc = binary_acc(pred, batch.label).item()
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

def eval(rnn, iterator, criterion):
    avg_acc = []
    rnn.eval()
    with torch.no_grad():
        for batch in iterator:
            # [b, 1] => [b]
            pred = rnn(batch.text).squeeze(1)
            loss = criterion(pred, batch.label)
            acc = binary_acc(pred, batch.label).item()
            avg_acc.append(acc)
    avg_acc = np.array(avg_acc).mean()
    print('>> test: ', avg_acc)


