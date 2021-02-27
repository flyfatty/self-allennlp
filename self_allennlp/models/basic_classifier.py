# @Time : 2021/2/20 15:26
# @Author : LiuBin
# @File : basic_classifier.py
# @Description : 
# @Software: PyCharm

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn import util, InitializerApplicator
from allennlp.models import Model, BasicClassifier
from allennlp.training.metrics import F1Measure
from allennlp.nn.util import get_text_field_mask


@Model.register('basic_classifier_f')  # 注册一个Model名称
class BasicClassifierF(BasicClassifier):
    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            seq2seq_encoder: Seq2SeqEncoder = None,
            feedforward: Optional[FeedForward] = None,
            dropout: float = None,
            num_labels: int = None,
            label_namespace: str = "labels",
            namespace: str = "tokens",
            initializer: InitializerApplicator = InitializerApplicator(),
            f_beta=1):
        super().__init__(vocab, text_field_embedder, seq2vec_encoder, seq2seq_encoder, feedforward, dropout, num_labels,
                         label_namespace, namespace, initializer)

        self._beta = f_beta
        self._f1 = F1Measure(1)

    def forward(self,
                tokens: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            self._f1(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        metrics.update(self._f1.get_metric(reset))
        return metrics
