# @Time : 2020/12/30 15:31
# @Author : LiuBin
# @File : generator.py
# @Description : 
# @Software: PyCharm
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # B*N*d_model x d_model*voc ==> B*N*voc
        return F.log_softmax(self.proj(x), dim=-1)
