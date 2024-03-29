# @Time : 2020/12/24 1:02
# @Author : LiuBin
# @File : bert_embedder.py
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

# Splits text into words (instead of wordpieces or characters).  For ELMo, you can
# just use any word-level tokenizer that you like, though for best results you
# should use the same tokenizer that was used with ELMo, which is an older version
# of spacy.  We're using a whitespace tokenizer here for ease of demonstration
# with binder.
tokenizer = WhitespaceTokenizer()

# Represents each token with an array of characters in a way that ELMo expects.
token_indexer = ELMoTokenCharactersIndexer()

# Both ELMo and BERT do their own thing with vocabularies, so we don't need to add
# anything, but we do need to construct the vocab object so we can use it below.
# (And if you have any labels in your data that need indexing, you'll still need
# this.)
vocab = Vocabulary()

text = "This is some text ."
tokens = tokenizer.tokenize(text)
print("ELMo tokens:", tokens)

text_field = TextField(tokens, {'elmo_tokens': token_indexer})
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print("ELMo tensors:", tensor_dict)

# Any transformer model name that huggingface's transformers library supports will
# work here.  Under the hood, we're grabbing pieces from huggingface for this
# part.
transformer_model = 'bert-base-cased'

# To do modeling with BERT correctly, we can't use just any tokenizer; we need to
# use BERT's tokenizer.
tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model)

# Represents each wordpiece with an id from BERT's vocabulary.
token_indexer = PretrainedTransformerIndexer(model_name=transformer_model)

text = "Some text with an extraordinarily long identifier."
tokens = tokenizer.tokenize(text)
print("BERT tokens:", tokens)

text_field = TextField(tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)

tensor_dict = text_field.as_tensor(text_field.get_padding_lengths())
print("BERT tensors:", tensor_dict)

# Now we'll do an example with paired text, to show the right way to handle [SEP]
# tokens in AllenNLP.  We have built-in ways of handling this for two text pieces.
# If you have more than two text pieces, you'll have to manually add the special
# tokens.  The way we're doing this requires that you use a
# PretrainedTransformerTokenizer, not the abstract Tokenizer class.

# Splits text into wordpieces, but without adding special tokens.
tokenizer = PretrainedTransformerTokenizer(
    model_name=transformer_model,
    add_special_tokens=False,
)

context_text = "This context is frandibulous."
question_text = "What is the context like?"
context_tokens = tokenizer.tokenize(context_text)
question_tokens = tokenizer.tokenize(question_text)
print("Context tokens:", context_tokens)
print("Question tokens:", question_tokens)

combined_tokens = tokenizer.add_special_tokens(context_tokens, question_tokens)
print("Combined tokens:", combined_tokens)

text_field = TextField(combined_tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)

tensor_dict = text_field.as_tensor(text_field.get_padding_lengths())
print("Combined BERT tensors:", tensor_dict)

# It's easiest to get ELMo input by just running the data code.  See the
# exercise above for an explanation of this code.
tokenizer = WhitespaceTokenizer()
token_indexer = ELMoTokenCharactersIndexer()
vocab = Vocabulary()
text = "This is some text."
tokens = tokenizer.tokenize(text)
print("ELMo tokens:", tokens)
text_field = TextField(tokens, {'elmo_tokens': token_indexer})
text_field.index(vocab)
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
print("ELMo tensors:", token_tensor)

# We're using a tiny, toy version of ELMo to demonstrate this.
elmo_options_file = 'https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/options.json'
elmo_weight_file = 'https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/lm_weights.hdf5'
elmo_embedding = ElmoTokenEmbedder(options_file=elmo_options_file,
                                   weight_file=elmo_weight_file)

embedder = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("ELMo embedded tokens:", embedded_tokens)

# Again, it's easier to just run the data code to get the right output.

# We're using the smallest transformer model we can here, so that it runs on
# binder.
transformer_model = 'google/reformer-crime-and-punishment'
tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model)
token_indexer = PretrainedTransformerIndexer(model_name=transformer_model)
text = "Some text with an extraordinarily long identifier."
tokens = tokenizer.tokenize(text)
print("Transformer tokens:", tokens)
text_field = TextField(tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
print("Transformer tensors:", token_tensor)

embedding = PretrainedTransformerEmbedder(model_name=transformer_model)

embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("Transformer embedded tokens:", embedded_tokens)

