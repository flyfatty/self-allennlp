# @Time : 2020/12/17 19:28
# @Author : LiuBin
# @File : model.py
# @Description : 
# @Software: PyCharm
from typing import Dict

import torch
import torch.nn.functional as F
import numpy
from allennlp.data import PyTorchDataLoader, Instance, Token, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.fields import TextField, LabelField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import Model, basic_classifier, simple_tagger
from allennlp.training.metrics import CategoricalAccuracy

#  The most important is the fact that forward() returns a dictionary, unlike most PyTorch Modules, which usually return a tensor.
# Create a toy model that just prints tensors passed to forward
class ToyModel(Model):
    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                tokens: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        print(tokens)
        print(label)
        # Simply generate random logits and compute a probability distribution
        batch_size = label.size(0)
        logits = torch.normal(mean=0., std=1., size=(batch_size, 2))
        probs = F.softmax(logits, dim=1)

        return {'logits': logits, 'probs': probs}

    def make_output_human_readable(self,
                                   output_dict: Dict[str, torch.Tensor]
                                   ) -> Dict[str, torch.Tensor]:
        # Take the logits from the forward pass, and compute the label
        # IDs for maximum values
        predicted_id = torch.argmax(output_dict['logits'], axis=-1)
        # Convert these IDs back to label strings using vocab
        output_dict['label'] = [
            self.vocab.get_token_from_index(x.item(), namespace='labels')
            for x in predicted_id
        ]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


# Create fields and instances
token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens')}
text_field_pos = TextField(
    [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')],
    token_indexers=token_indexers)
text_field_neg = TextField(
    [Token('Such'), Token('an'), Token('awful'), Token('movie'), Token('.')],
    token_indexers=token_indexers)

label_field_pos = LabelField('pos', label_namespace='labels')
label_field_neg = LabelField('neg', label_namespace='labels')

instance_pos = Instance({'tokens': text_field_pos, 'label': label_field_pos})
instance_neg = Instance({'tokens': text_field_neg, 'label': label_field_neg})
instances = [instance_pos, instance_neg]

# Create a Vocabulary
vocab = Vocabulary.from_instances(instances)

dataset = AllennlpDataset(instances, vocab)

# Create an iterator that creates batches of size 2
data_loader = PyTorchDataLoader(dataset, batch_size=2)

model = ToyModel(vocab)

# Iterate over batches and pass them to forward()
for batch in data_loader:
    a = model(**batch)
    print(a)
# if you want to train your model through backpropagation using our Trainer, the return value must contain a "loss" key pointing to a scalar Tensor

# Run forward pass on an instance. This will invoke forward() then decode()
print(model.forward_on_instance(instance_pos))
# Notice that the return value is one dictionary per instance,
# even though everything in forward() and decode() is batched
print(model.forward_on_instances([instance_pos, instance_neg]))

print(model.get_metrics())
