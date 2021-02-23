# @Time : 2021/2/20 15:26
# @Author : LiuBin
# @File : simple_classifier.py
# @Description : 
# @Software: PyCharm

from typing import Dict

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register('simple_classifier')  # 注册一个Model名称
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,  # The final constructed vocab gets passed to the model automatically.
                 embedder: TextFieldEmbedder,
                 # recommend to simply take a TextFieldEmbedder as a constructor parameter .
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(1)

    def forward(self,
                tokens: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = F.softmax(logits)
        # Shape: (1,)
        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            self.f1(logits, label)
            output['loss'] = F.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self.accuracy.get_metric(reset)}
        metrics.update(self.f1.get_metric(reset))
        return metrics
