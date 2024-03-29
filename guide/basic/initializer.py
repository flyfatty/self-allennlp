# @Time : 2020/12/18 14:30
# @Author : LiuBin
# @File : initializer.py
# @Description : 
# @Software: PyCharm

import torch
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.nn.initializers import InitializerApplicator, XavierUniformInitializer, ConstantInitializer, \
    NormalInitializer


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 2)
        self.conv = torch.nn.Conv1d(2, 2, 2)

    def forward(self, inputs):
        pass


model = Net()
print('Initial parameters:')
for name, param in model.named_parameters():
    print(name, param)

init_uniform = XavierUniformInitializer()
init_uniform(model.linear1.weight)
init_uniform(model.linear2.weight)

init_const = ConstantInitializer(val=10.)
init_const(model.linear1.bias)
init_const(model.linear2.bias)

init_normal = NormalInitializer(mean=0., std=10.)
init_normal(model.conv.weight)
init_normal(model.conv.bias)

print('\nAfter applying initializers individually:')
for name, param in model.named_parameters():
    print(name, param)

model = Net()
applicator = InitializerApplicator(
    regexes=[
        ('linear.*weight', init_uniform),
        ('linear.*bias', init_const),
        ('conv.*', init_normal)
    ])
applicator(model)

print('\nAfter applying an applicator:')
for name, param in model.named_parameters():
    print(name, param)

# This can be achieved by adding the following two lines to the model constructor:
class YourModel(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            initializer: InitializerApplicator = InitializerApplicator()
    ) -> None:
        super().__init__(vocab)
        initializer(self)
