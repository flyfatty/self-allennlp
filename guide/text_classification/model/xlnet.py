import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetConfig, XLNetTokenizer, XLNetForSequenceClassification

class XLNetForTextClassification(nn.Module):
    def __init__(self, pretrained_model_path, num_classes, d_model=1024, n_layer=24, n_head=16, \
            d_inner=4096, ff_activation='gelu', untie_r=True, attn_type='bi',initializer_range=0.02, \
            layer_norm_eps=1e-12, dropout=0.1, use_pretrained_model=False):
        super(XLNetForTextClassification, self).__init__()

        if(use_pretrained_model==True):
            print('Reloading pretrained models...')
            self.model = XLNetForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=num_classes)
        else:
            print('Constructing new xlnet by parameters...')
            tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_path)
            vocab_size = len(tokenizer.ids_to_tokens)

            config = XLNetConfig(vocab_size, d_model, n_layer, n_head, \
                                d_inner, ff_activation, untie_r, attn_type, \
                                initializer_range, layer_norm_eps, dropout)
            self.model = XLNetForSequenceClassification(config)

    def forward(self, input_senquence, batch_size=None):
        '''
        Args:
        input_senquence: shape=[batch_size, max_time_step]

        Returns:
        logits: shape=[batch_size, num_labels]
        '''
        logits = self.model(input_senquence)[0]
        return logits