# @Time : 2020/12/18 16:46
# @Author : LiuBin
# @File : __init__.py.py
# @Description : 
# @Software: PyCharm

from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, Conll2003DatasetReader, ShardedDatasetReader, \
    InterleavingDatasetReader, AllennlpLazyDataset
from .dataset_readers import *
from .tokenizers import *
