# @Time : 2020/12/22 15:14
# @Author : LiuBin
# @File : 3.indexer.py
# @Description : 
# @Software: PyCharm

from allennlp.data import Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer
)
from allennlp.data.tokenizers import (
    CharacterTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)

# 第一个例子，word分词+singleID索引
tokenizer = WhitespaceTokenizer()

# 建立索引器时设置对应的词表空间，默认"tokens"
token_indexer = SingleIdTokenIndexer(namespace='token_vocab')

# 手动构造词表，指定词表空间
vocab = Vocabulary()
vocab.add_tokens_to_namespace(['This', 'is', 'some', 'text', '.'], namespace='token_vocab')
vocab.add_tokens_to_namespace(['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'], namespace='character_vocab')

# 构造一个样本
text = "This is some text ."
tokens = tokenizer.tokenize(text)
print("Word tokens:", tokens)

# 指定TextField时指定Indexer包，key作为索引空间。
# 索引空间和词表空间的区别：索引空间与Embedding层一一对应，不同的索引空间学习不同的Embedding矩阵。不同索引空间、同一词表空间表示可以共享词表，但不共享Embedding矩阵
text_field = TextField(tokens, {'tokens': token_indexer})

# 用词表生成Field对应的索引序列，结果维护在 indexed_tokens
text_field.index(vocab)

# 获得每个"索引空间_key"的长度
padding_lengths = text_field.get_padding_lengths()

# 生成 indexed tensor 准备属入到Model
tensor_dict = text_field.as_tensor(padding_lengths)
print("With single id indexer:", tensor_dict)

# 第二个例子，，word分词+char索引, 使用另一个Indexer,指定词表空间
token_indexer = TokenCharactersIndexer(namespace='character_vocab')

# 建立Field，维护在 索引空间 token_characters
text_field = TextField(tokens, {'token_characters': token_indexer})
# 利用词表和索引器生成索引序列维护在Field ， 一个Token --> 一个索引序列
text_field.index(vocab)

# 获得padding后的长度，后面用来构造tensor
padding_lengths = text_field.get_padding_lengths()
print("padding_lengths ", padding_lengths)

# padding 并 转换 tensor
tensor_dict = text_field.as_tensor(padding_lengths)
# 生成Tensor格式 [B x N x C]
print("With token characters indexer:", tensor_dict)

# 第三个例子：char 分词器 + SingleID索引器
# char 分词器(instead of words or wordpieces).
tokenizer = CharacterTokenizer()

tokens = tokenizer.tokenize(text)
print("Character tokens:", tokens)

# 建立Single ID索引器，指定词表空间
token_indexer = SingleIdTokenIndexer(namespace='character_vocab')
text_field = TextField(tokens, {'token_characters': token_indexer})
text_field.index(vocab)

padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print("With single id indexer:", tensor_dict)

# 第四个例子，word分词+多个索引器
tokenizer = WhitespaceTokenizer()

# word-->idx 和 word --> idx 序列 ， 指定词表空间
token_indexers = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
    'token_characters': TokenCharactersIndexer(namespace='character_vocab')
}

text = "This is some text ."
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# 建立Field ， 配置 indexer字典
text_field = TextField(tokens, token_indexers)
text_field.index(vocab)

padding_lengths = text_field.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
# 输出 索引空间+索引类型+Tensor的字典 Dict[str,Dict[str,Tensor]]
print("Combined tensor dictionary:", tensor_dict)

# 第五个例子，使用Spacy库获得POS Tag，保存在Token.tag_
tokenizer = SpacyTokenizer(language="en_core_web_sm", pos_tags=True)
# 词表中添加 新的词表空间
vocab.add_tokens_to_namespace(['DT', 'VBZ', 'NN', '.'], namespace='pos_tag_vocab')
# 建立三个索引器，分配索引空间和设置词表空间，其中两个single id索引器和一个char索引器
token_indexers = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
    'token_characters': TokenCharactersIndexer(namespace='character_vocab'),
    'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab', feature_name='tag_'),
}
tokens = tokenizer.tokenize(text)
print("Token tags:", [token.text for token in tokens], "POS tags:", [token.tag_ for token in tokens])

text_field = TextField(tokens, token_indexers)
text_field.index(vocab)

padding_lengths = text_field.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
# 生成key分别是tokens、token_character、pos_tags索引空间
# 对应的值(内部字典)是key为tokens、token_characters、tokens索引类型的Tensor
print("Tensor dict with POS tags:", tensor_dict)
