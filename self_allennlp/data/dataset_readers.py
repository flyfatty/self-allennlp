# @Time : 2020/12/18 16:16
# @Author : LiuBin
# @File : dataset_readers.py
# @Description : 
# @Software: PyCharm
"""
通用 DatasetReader
任务类型: 分类
分隔符: Tab
输入: 序列Sequence
输出: 单个Label
"""
import logging
import numpy as np
from overrides import overrides
from typing import Iterable, Dict, List
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import TextField, LabelField, ListField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer

from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp_models.generation.dataset_readers import Seq2SeqDatasetReader
from allennlp_models.pair_classification.dataset_readers import QuoraParaphraseDatasetReader

"""
## 内置
sequence_tagging  序列标注

## 内置扩展
seq2seq    生成任务      
quora_paraphrase

## 自定义
cls_tsv_dataset_reader    分类任务

## 定制
mimic    QA
"""


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("cls_tsv_dataset_reader")
class ClsTsvDataSetReader(DatasetReader):

    def __init__(self, tokenizer: Tokenizer = None,
                 token_indexer: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 label_first: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexer = token_indexer or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.label_first = label_first

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, label = line.strip().split('\t')
                if self.label_first:
                    label, text = text, label
                yield self.text_to_instance(text, label)

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_fields = TextField(tokens, self.token_indexer)
        fields = {"tokens": text_fields}
        if label:
            label_fields = LabelField(label)
            fields["label"] = label_fields
        return Instance(fields)


# @DatasetReader.register("cls_folder_dataset_reader")
# class ClsFolderDataSetReader(DatasetReader):
#     def __init__(self, tokenizer: Tokenizer = None,
#                  token_indexer: Dict[str, TokenIndexer] = None,
#                  max_tokens: int = None,
#                  **kwargs) -> None:
#
#     def _read(self, file_path: str) -> Iterable[Instance]:
#
#     def text_to_instance(self, *inputs) -> Instance:
#         pass

#####################################################
#           定制Dataset Reader
#
#
#
#
#####################################################
@DatasetReader.register("mimics")
class MIMICSDatasetReader(DatasetReader):

    def _read(self, file_path: str) -> Iterable[Instance]:
        pass

    @overrides
    def text_to_instance(
            self,
            query: str,
            question: str,
            options: List[str],
            labels: List[float] = None
    ) -> Instance:

        token_field = self._make_textfield(' [SEP] '.join([query, question]))
        options_field = self._make_listfield(options)

        fields = {'tokens': token_field, 'options': options_field}
        if labels:
            fields['labels'] = ArrayField(np.array(labels), padding_value=-1)

        return Instance(fields)

    def _make_textfield(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return TextField(tokens, token_indexers=self.token_indexers)

    def _make_listfield(self, documents: List[str]):
        return ListField([self._make_textfield(d) for d in documents])


if __name__ == '__main__':
    MIMICSDatasetReader()