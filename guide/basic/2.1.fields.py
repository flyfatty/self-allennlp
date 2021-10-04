# @Time : 2020/12/16 16:45
# @Author : LiuBin
# @File : 2.1.fields.py
# @Description : 
# @Software: PyCharm

from collections import Counter, defaultdict

from allennlp.data.fields import LabelField, SequenceLabelField

import torch
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
)
from allennlp.data.tokenizers import (
    SpacyTokenizer
)
from allennlp.nn import util as nn_util

# To create fields, simply pass the data to constructor.
# NOTE: Don't worry about the token_indexers too much for now. We have a whole
# chapter on why TextFields are set up this way, and how they work.
tokens = [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')]
token_indexers = {'tokens': SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

label_field = LabelField('pos')

sequence_label_field = SequenceLabelField(
    ['DET', 'ADJ', 'NOUN', 'ADV', 'PUNKT'],
    text_field
)

# You can use print() fields to see their content
print(text_field)
print(label_field)
print(sequence_label_field)

# Many of the fields implement native python methods in intuitive ways
print(len(sequence_label_field))
print(label for label in sequence_label_field)

# Fields know how to create empty fields of the same type
print(text_field.empty_field())
print(label_field.empty_field())
print(sequence_label_field.empty_field())

# You can count vocabulary items in fields
counter = defaultdict(Counter)
text_field.count_vocab_items(counter)
print(counter)

label_field.count_vocab_items(counter)
print(counter)

sequence_label_field.count_vocab_items(counter)
print(counter)

# Create Vocabulary for indexing fields
vocab = Vocabulary(counter)

# Fields know how to turn themselves into tensors
text_field.index(vocab)
# NOTE: in practice, we will batch together instances and use the maximum padding
# lengths, instead of getting them from a single instance.
# You can print this if you want to see what the padding_lengths dictionary looks
# like, but it can sometimes be a bit cryptic.
padding_lengths = text_field.get_padding_lengths()
print(text_field.as_tensor(padding_lengths))

label_field.index(vocab)
print(label_field.as_tensor(label_field.get_padding_lengths()))

sequence_label_field.index(vocab)
padding_lengths = sequence_label_field.get_padding_lengths()
print(sequence_label_field.as_tensor(padding_lengths))

# Fields know how to batch tensors
tensor1 = label_field.as_tensor(label_field.get_padding_lengths())

label_field2 = LabelField('pos')
label_field2.index(vocab)
tensor2 = label_field2.as_tensor(label_field2.get_padding_lengths())

batched_tensors = label_field.batch_tensors([tensor1, tensor2])
print(batched_tensors)



# We're following the logic from the "Combining multiple TokenIndexers" example
# above.
tokenizer = SpacyTokenizer(pos_tags=True)

vocab = Vocabulary()
vocab.add_tokens_to_namespace(
    ['This', 'is', 'some', 'text', '.'],
    namespace='token_vocab')
vocab.add_tokens_to_namespace(
    ['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'],
    namespace='character_vocab')
vocab.add_tokens_to_namespace(['DT', 'VBZ', 'NN', '.'],
                              namespace='pos_tag_vocab')

text = "This is some text."
text2 = "This is some text with more tokens."
tokens = tokenizer.tokenize(text)
tokens2 = tokenizer.tokenize(text2)
print("Tokens:", tokens)
print("Tokens 2:", tokens2)


# Represents each token with (1) an id from a vocabulary, (2) a sequence of
# characters, and (3) part of speech tag ids.
token_indexers = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
    'token_characters': TokenCharactersIndexer(namespace='character_vocab'),
    'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab',
                                     feature_name='tag_'),
}

text_field = TextField(tokens, token_indexers)
text_field.index(vocab)
text_field2 = TextField(tokens2, token_indexers)
text_field2.index(vocab)

# We're using the longer padding lengths here; we'd typically be relying on our
# collate function to figure out what the longest values are to use.
padding_lengths = text_field2.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
tensor_dict2 = text_field2.as_tensor(padding_lengths)
print("Combined tensor dictionary:", tensor_dict)
print("Combined tensor dictionary 2:", tensor_dict2)

text_field_tensors = text_field.batch_tensors([tensor_dict, tensor_dict2])
print("Batched tensor dictionary:", text_field_tensors)

# We've seen plenty of examples of using a TextFieldEmbedder, so we'll just show
# the utility methods here.
mask = nn_util.get_text_field_mask(text_field_tensors)
print("Mask:", mask)
print("Mask size:", mask.size())
token_ids = nn_util.get_token_ids_from_text_field_tensors(text_field_tensors)
print("Token ids:", token_ids)

# We can also handle getting masks when you have lists of TextFields, but there's
# an important parameter that you need to pass, which we'll show here.  The
# difference in output that you see between here and above is just that there's an
# extra dimension in this output.  Where shapes used to be (batch_size=2, ...),
# now they are (batch_size=1, list_length=2, ...).
list_field = ListField([text_field, text_field2])
tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
text_field_tensors = list_field.batch_tensors([tensor_dict])
print("Batched tensors for ListField[TextField]:", text_field_tensors)

# The num_wrapping_dims argument tells get_text_field_mask how many nested lists
# there are around the TextField, which we need for our heuristics that guess
# which tensor to use when computing a mask.
mask = nn_util.get_text_field_mask(text_field_tensors, num_wrapping_dims=1)
print("Mask:", mask)
print("Mask:", mask.size())