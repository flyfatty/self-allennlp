# @Time : 2021/1/8 10:57
# @Author : LiuBin
# @File : lstm.py
# @Description : 
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch, target_batch = [], []

    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
        target = word_dict[seq[-1]] # 'e' is target
        input_batch.append(np.eye(n_class)[input]) # 造embedding向量
        target_batch.append(target)

    return input_batch, target_batch

class TextLSTM(nn.Module):
    """
    LSTM + Linear分类器
    """
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden,batch_first=True)
        self.W = nn.Linear(n_hidden, n_class, bias=True)

    def forward(self, X):

        outputs, (_, _) = self.lstm(X)
        outputs = outputs[:,-1]  # B*N*H --> B*H
        model = self.W(outputs)   # model :B*H x H*O + O --> B*O
        return model

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        # 因为 word embedding 是 one-hot编码，所以 input-size是 n_class
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True,batch_first=True)
        self.W = nn.Linear(n_hidden * 2, n_class, bias=True)

    def forward(self, X):

        outputs, (_, _) = self.lstm(X)
        outputs = outputs[:,-1]  # B*(H+H)   Concat 双向向量
        model = self.W(outputs)  # model : [batch_size, n_class]
        return model

if __name__ == '__main__':
    n_hidden = 128 # number of hidden units in one cell

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n: i for i, n in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    n_class = len(word_dict)  # number of class(=number of vocab)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    model = TextLSTM()
    model2 = BiLSTM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(1000):
        optimizer.zero_grad()

        output = model2(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    inputs = [sen[:3] for sen in seq_data]

    predict = model2(input_batch).data.max(1)[1]

    print(inputs, '->', [number_dict[n.item()] for n in predict])