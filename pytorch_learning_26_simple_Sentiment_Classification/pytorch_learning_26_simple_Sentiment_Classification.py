# pytorch_learning_26_simple_Sentiment_Classification

# Google CoLab
# K80 可以使用12h 会清洗

import torch
import torch.nn as nn
from torch import optim
from torchtext import data

TEXT = data.Field(tokensize='spacy')
LABEL = data.LabelField(dtype=torch.float)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.embbeding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)

