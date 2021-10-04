# @Time : 2020/12/17 11:53
# @Author : LiuBin
# @File : 1.dateset_readers.py
# @Description : 
# @Software: PyCharm
import os
from itertools import islice
from typing import Dict, Iterable

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

from config import ConfigManager

config = ConfigManager()

# 继承 DatasetReader 并 _read() 和 text_to_instance()
@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                yield self.text_to_instance(text, sentiment)


# Instantiate and use the dataset reader to read a file containing the data
reader = ClassificationTsvReader(cache_directory='dataset_cache')
dataset = reader.read(os.path.join(config.DATA_PATH, 'movie_review/train.tsv'))    # AllennlpDataset

# Returned dataset is a list of Instances by default  #  jsonpickle
print('type of dataset: ', type(dataset))
print('type of its first element: ', type(dataset[0]))
print('size of dataset: ', len(dataset))
# Check if the dataset is cached
print(os.listdir('dataset_cache'))

# lazy read mode
reader = ClassificationTsvReader(lazy=True)
dataset = reader.read(os.path.join(config.DATA_PATH, 'movie_review/train.tsv'))
print('type of dataset: ', type(dataset))
print('first 5 instances: ', list(islice(dataset, 5)))
