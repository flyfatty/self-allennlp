# @Time : 2020/12/24 14:46
# @Author : LiuBin
# @File : pooled_embedder.py
# @Description : 
# @Software: PyCharm

import torch
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.tokenizers import (
    CharacterTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)
from allennlp.nn import util as nn_util

import warnings
warnings.filterwarnings("ignore")


# 准备好 word-level 分词 ，不用多说
text_tokens = ["This", "is", "some", "frandibulous", "text", "."]
tokens = [Token(x) for x in text_tokens]
print(tokens)

# 用一个 tiny transformer测试，也可以指定任何transformer包的模型
transformer_model = 'google/reformer-crime-and-punishment'

# Represents the list of word tokens with a sequences of wordpieces as determined
# by the transformer's tokenizer.  This actually results in a pretty complex data
# type, which you can see by running this.  It's complicated because we need to
# know how to combine the wordpieces back into words after running the
# transformer.
indexer = PretrainedTransformerMismatchedIndexer(model_name=transformer_model)

text_field = TextField(tokens, {'transformer': indexer})
text_field.index(Vocabulary())
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())

# There are two key things to notice in this output.  First, there are two masks:
# `mask` is a word-level mask that gets used in the utility functions described in
# the last section of this chapter.  `wordpiece_mask` gets used by the `Embedder`
# itself.  Second, there is an `offsets` tensor that gives start and end wordpiece
# indices for the original tokens.  In the embedder, we grab these, average all of
# the wordpieces for each token, and return the result.
print("Indexed tensors:", token_tensor)

embedding = PretrainedTransformerMismatchedEmbedder(model_name=transformer_model)

embedder = BasicTextFieldEmbedder(token_embedders={'transformer': embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("Embedded tokens size:", embedded_tokens.size())
print("Embedded tokens:", embedded_tokens)