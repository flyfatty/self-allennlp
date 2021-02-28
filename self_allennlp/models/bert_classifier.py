# @Time : 2021/2/25 21:18
# @Author : LiuBin
# @File : bert_classifier.py
# @Description : 
# @Software: PyCharm
from typing import Dict
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.modules import TextFieldEmbedder


@Model.register('bert_classifier')
class BertClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 freeze_encoder: bool = True
                 ):
        super().__init__(vocab)
        self.embedder = embedder
        self.freeze_encoder = freeze_encoder
        for parameter in self.embedder.parameters():
            parameter.requires_grad = not self.freeze_encoder
        self._linear = torch.nn.Linear(in_features=self.embedder.get_output_dim(),
                                       out_features=vocab.get_vocab_size("labels"))

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # B x max_len x embedding_dim
        embeddings = self.embedder(tokens)
        # B x embedding_dim
        cls_embedding = embeddings[:, 0, :]
        # B x num_labels
        logits = self._linear(cls_embedding)
        output_dict = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        if label is not None:
            self.accuracy(logits, label)
            output_dict["loss"] = self._loss(logits, label)
        return output_dict

    def get_metrics(self, reset=False):
        return {'accuracy': self._accuracy.get_metric(reset)}
