# @Time : 2020/12/24 14:23
# @Author : LiuBin
# @File : pretrain_embedding.py
# @Description : 
# @Software: PyCharm

from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
import torch

# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer;
# see the exercises above.
token_tensor = {'tokens': {'tokens': torch.LongTensor([1, 3, 2, 1, 4, 3])}}

vocab = Vocabulary()
vocab.add_tokens_to_namespace(['This', 'is', 'some', 'text', '.'],
                              namespace='token_vocab')

glove_file = '/home/liubin/tutorials/pytorch/self-allennlp/data/embedding/glove.6B.50d.txt'

# This is for embedding each token.
embedding = Embedding(vocab=vocab,
                      vocab_namespace='token_vocab',
                      embedding_dim=50,
                      pretrained_file=glove_file)

embedder = BasicTextFieldEmbedder(token_embedders={'tokens': embedding})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens.size())