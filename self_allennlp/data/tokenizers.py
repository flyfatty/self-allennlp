# @Time : 2020/12/18 22:55
# @Author : LiuBin
# @File : tokenizers.py
# @Description : 
# @Software: PyCharm
import os
import jieba.posseg as poss
import jieba

from typing import List
from overrides import overrides
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.token_class import show_token

from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.tokenizers import LettersDigitsTokenizer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


@Tokenizer.register('jieba')
class JiebaTokenizer(Tokenizer):
    """
    A ``Tokenizer`` that uses JIEBA's tokenizer. To Split Chinese sentences.
    user_dict:a txt file, one word in a line.
    """

    def __init__(self, pos_tags: bool = False,
                 user_dict: str = None) -> None:
        self._pos_tags = pos_tags

        if user_dict and os.path.exists(user_dict):
            jieba.load_userdict(user_dict)

        self.tokenizer = poss if pos_tags else jieba

    def _sanitize(self, tokens) -> List[Token]:
        """
        Converts spaCy tokens to allennlp tokens. Is a no-op if
        keep_spacy_tokens is True
        """
        sanitize_tokens = []
        if self._pos_tags:
            for text, pos in tokens:
                token = Token(text, pos_=pos)
                sanitize_tokens.append(token)
        else:
            for token in tokens:
                token = Token(token)
                sanitize_tokens.append(token)
        return sanitize_tokens

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self._sanitize(tokens) for sent in texts for tokens in self.tokenizer.cut(sent)]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return self._sanitize(self.tokenizer.cut(text))

    def show_tokens(self, tokens: List[Token]):
        for token in tokens:
            print(show_token(token))


if __name__ == '__main__':
    tokenizer = JiebaTokenizer(pos_tags=True)
    print(tokenizer.show_tokens(tokenizer.tokenize("我爱北京天安门！天安门上太阳升起。。")))
