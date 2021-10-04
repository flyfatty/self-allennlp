import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class RobertaForTextClassification(nn.Module):
    def __init__(self, pretrained_model_path, num_classes, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, \
            intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, 
            max_position_embeddings=512, use_pretrained_model=False):
        # super(RobertaForSequenceClassification, self).__init__()
        # super(AutoModelForSequenceClassification, self).__init__()
        super(RobertaForTextClassification, self).__init__()
        # config = AutoConfig.from_pretrained(pretrained_model_path)
        # setattr(config, 'num_labels', num_classes)
        if(use_pretrained_model==True):
            print('Reloading pretrained models...')
            self.model = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=num_classes)
            # self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes)
            # self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, config=config)
        else:
            print('Constructing new roberta by parameters...')
            tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
            vocab_size = len(tokenizer.ids_to_tokens)
            config = RobertaConfig(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, \
                    intermediate_size, hidden_act, hidden_dropout_prob, \
                    attention_probs_dropout_prob, max_position_embeddings)
            self.model = RobertaForSequenceClassification(config)

    def forward(self, input_senquence, batch_size=None):
        '''
        Args:
        input_senquence: shape=[batch_size, max_time_step]

        Returns:
        logits: shape=[batch_size, num_labels]
        '''
        logits = self.model(input_senquence)[0]
        return logits

def one_data(text):
    pretrained_model_path = '/share/fwp/tools/auto_text_classifier/atc/data/chinese_roberta_wwm_ext'
    model_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    sequence = model_tokenizer.encode(text, add_special_tokens=True)
    print('text', text)
    print('sequence', sequence)
    

if __name__ == "__main__":
    import os
    print(1)
    one_data('我试试一下这个')
    print(2)
    one_data('我 试试 一下 这个')

