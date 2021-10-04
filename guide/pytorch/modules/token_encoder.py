# @Time : 2021/3/23 12:22
# @Author : LiuBin
# @File : token_encoder.py
# @Description : 
# @Software: PyCharm
import math
import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # 生成空位置向量   max_len*d_model
        position = torch.arange(0, max_len).unsqueeze(1)  # max_len*1
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1*max_len*d_model
        self.register_buffer('pe', pe)  # 设置为buffer（不需要求梯度的参数）

    def forward(self, x):
        # B*N*d_model + 1*N*d_model --> B*N*d_model
        x = x + torch.tensor(self.pe[:, :x.size(1)])
        return self.dropout(x)
