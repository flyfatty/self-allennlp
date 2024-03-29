# @Time : 2020/12/17 15:50
# @Author : LiuBin
# @File : 2.3.vocabulary.py
# @Description : 
# @Software: PyCharm

from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

# By default, namespaces ending in "tags" or "labels" are treated as non-padded
# Create fields and instances

# We will use the namespace 'tokens' to map tokens to integers.  This is the
# default value, but we are passing it here explicitly to make it clear.
token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens')}
text_field_pos = TextField(
    [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')],
    token_indexers=token_indexers)
text_field_neg = TextField(
    [Token('Such'), Token('an'), Token('awful'), Token('movie'), Token('.')],
    token_indexers=token_indexers)

# Similarly, we will use the default namespace 'labels' to map labels to integers.
label_field_pos = LabelField('pos', label_namespace='labels')
label_field_neg = LabelField('neg', label_namespace='labels')

instance_pos = Instance({'tokens': text_field_pos, 'label': label_field_pos})
instance_neg = Instance({'tokens': text_field_neg, 'label': label_field_neg})

# Create a Vocabulary

# Tokens from text fields are managed by the 'tokens' namespace, while
# labels are stored under the `labels` namespace, as we specified above.
vocab = Vocabulary.from_instances([instance_pos, instance_neg])

print('Created a Vocabulary:', vocab)

# Looking up indices. namespace='tokens' is used by default.  It turns out that we
# add vocabulary items in frequency order, so "movie", which showed up twice, will
# get the lowest index, other than padding and OOV.
print('index for token "movie":', vocab.get_token_index('movie'))
print('index for token "!":', vocab.get_token_index('!'))
# 'tokens' is a padded namespace, and unknown tokens get mapped to @@UNKNOWN@@
# (index = 1).
print('index for token "unknown":', vocab.get_token_index('unknown'))
print('index for token "music":', vocab.get_token_index('music'))

print('index for label "pos":', vocab.get_token_index('pos', namespace='labels'))
print('index for label "neg":', vocab.get_token_index('neg', namespace='labels'))

# 'labels' is a non-padded namespace; looking up unknown labels throws an error
try:
    vocab.get_token_index('unknown', namespace='labels')
except KeyError:
    print('index for label "unknown": caught KeyError')

# Looking up tokens and labels by indices

# Notice that for padded namespaces, '@@PADDING@@' and '@@UNKNOWN@@' are
# automatically added.
print('token for index=0:', vocab.get_token_from_index(0))
print('token for index=1:', vocab.get_token_from_index(1))
print('token for index=2:', vocab.get_token_from_index(2))

print('label for index=0:', vocab.get_token_from_index(0, namespace='labels'))
print('label for index=1:', vocab.get_token_from_index(1, namespace='labels'))

try:
    vocab.get_token_from_index(2, namespace='labels')
except KeyError:
    print('label for index=2: caught KeyError')

# Create a Vocabulary with min_count=2 for tokens, but not for labels
vocab = Vocabulary.from_instances([instance_pos, instance_neg], min_count={'tokens': 2})

print('Created a Vocabulary:', vocab)

# Getting the entire mapping for "tokens." Notice only 'movie' is in the
# vocabulary.
print('index-to-token for "tokens":', vocab.get_index_to_token_vocabulary(namespace='tokens'))

# Getting the entire mapping for "labels." All the tokens that appeared in the
# dataset are present
print('index-to-token for "labels":', vocab.get_index_to_token_vocabulary(namespace='labels'))
