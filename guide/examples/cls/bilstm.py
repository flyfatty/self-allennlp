# @Time : 2021/1/8 13:42
# @Author : LiuBin
# @File : bilstm.py
# @Description : 
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input)) # padding
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input]) # one-hot for seq
        target_batch.append(target)

    return input_batch, target_batch

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

    sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )
    # 造 词表
    word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
    number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}

    n_class = len(word_dict)
    max_len = len(sentence.split())

    model = BiLSTM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1)[1]
    print(sentence)
    print([number_dict[n.item()] for n in predict])