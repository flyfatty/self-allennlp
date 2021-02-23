#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: test_model.py
@Software: PyCharm
@Time: 2021/2/8 11:43 上午
@Desc:

"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# 定义网络
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layer(x)


model = TestModel()

writer = SummaryWriter()
writer.add_graph(model, input_to_model=torch.randn((3, 3)))
writer.add_scalar(tag="test", scalar_value=torch.tensor(1), global_step=1)
writer.close()
