# @Time : 2020/12/22 16:34
# @Author : LiuBin
# @File : 4.embedders.py
# @Description : 
# @Software: PyCharm

import torch
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder
)
from allennlp.data import Vocabulary
import warnings

warnings.filterwarnings("ignore")

# token_tensor 等价于 Model forward的输入 text ， 格式为 Dict[str,Dict[str,Tensor]]
# Tensor的格式一般为为 [B x N] （word index） 或 [B x N x C] ( char index)

# 第一个key表示索引空间名称（自定义），第二个key表示来源于Indexer的类型（自动生成）
token_tensor = {'indexer1': {'tokens': torch.LongTensor([[1, 3, 2, 9, 4, 3]])}}

# 配置框架会自动根据Vocab定义num_embeddings，只需要在jsonnet中定义embedding_dim
embedding = Embedding(num_embeddings=10, embedding_dim=3)

# 把定义好的 token embedders放在TextFieldEmbedder里，维护在对应的索引空间中
embedder = BasicTextFieldEmbedder(token_embedders={'indexer1': embedding})

# 执行得到结果 ，格式为 B x N x D
embedded_tokens = embedder(token_tensor)
print("Using the TextFieldEmbedder:", embedded_tokens)

# 以上就是一般流程，如果就一个索引空间也可以不给它打包，直接使用 token embedder
embedded_tokens = embedding(**token_tensor['indexer1'])
print("Using the Embedding directly:", embedded_tokens)

# 这里是假设有两个索引空间 ，分别是 index1  [B x N] ， index2 [B x N x C]
token_tensor = {'indexer2': {'token_characters': torch.tensor([[[1, 3, 0], [4, 2, 3], [1, 9, 5], [6, 0, 0]]])},
                'indexer1': {'tokens': torch.LongTensor([[1, 3, 2, 9]])}}

# 定义一个token embedder对应 index1索引空间
embedding = Embedding(num_embeddings=10, embedding_dim=3)

# 定义另一个token embedder对饮 index2索引空间
character_embedding = Embedding(num_embeddings=10, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=(3,))
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# 将两个token_embedder按照对应的索引空间打包
embedder = BasicTextFieldEmbedder(token_embedders={'indexer2': token_encoder, 'indexer1': embedding})

# 执行得到embedding结果 , 格式为  [B x N x D_concat]
embedded_tokens = embedder(token_tensor)
print("With a character CNN:", embedded_tokens)

token_tensor = {
    'tokens': {'tokens': torch.LongTensor([[2, 4, 3, 5]])},
    'token_characters': {'token_characters': torch.LongTensor(
        [[[2, 5, 3], [4, 0, 0], [2, 1, 4], [5, 4, 0]]])}
}

# 再举一个例子，三个索引空间，使用了两个类型的Indexer
token_tensor = {
    'tokens': {'tokens': torch.LongTensor([[2, 4, 3, 5]])},
    'token_characters': {'token_characters': torch.LongTensor([[[2, 5, 3], [4, 0, 0], [2, 1, 4], [5, 4, 0]]])},
    'pos_tag_tokens': {'tokens': torch.tensor([[2, 5, 3, 4]])}
}

# 手动构造一个词表，指定词表空间。该过程由框架自动完成
vocab = Vocabulary()
vocab.add_tokens_to_namespace(['This', 'is', 'some', 'text', '.'], namespace='token_vocab')
vocab.add_tokens_to_namespace(['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'], namespace='character_vocab')
vocab.add_tokens_to_namespace(['DT', 'VBZ', 'NN', '.'], namespace='pos_tag_vocab')

# 建立Embedding层时指定词表及词表空间，也是配置启动时的过程，这样无需指定 num_embeddings参数
embedding = Embedding(embedding_dim=3, vocab_namespace='token_vocab', vocab=vocab)

character_embedding = Embedding(embedding_dim=4, vocab_namespace='character_vocab', vocab=vocab)
cnn_encoder = CnnEncoder(embedding_dim=4, num_filters=5, ngram_filter_sizes=[3])
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

pos_tag_embedding = Embedding(embedding_dim=6, vocab_namespace='pos_tag_vocab', vocab=vocab)

# 按照索引空间打包Token Embedder
embedder = BasicTextFieldEmbedder(
    token_embedders={'tokens': embedding,
                     'token_characters': token_encoder,
                     'pos_tag_tokens': pos_tag_embedding})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)
