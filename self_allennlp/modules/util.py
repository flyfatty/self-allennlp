# @Time : 2020/12/24 23:23
# @Author : LiuBin
# @File : util.py
# @Description : 
# @Software: PyCharm

from copy import deepcopy
from torch.nn import ModuleList


def clones(module, n):
    return ModuleList([deepcopy(module) for _ in range(n)])
