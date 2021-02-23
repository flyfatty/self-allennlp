# @Time : 2020/12/21 14:19
# @Author : LiuBin
# @File : __init__.py.py
# @Description : 
# @Software: PyCharm


from allennlp.modules.seq2vec_encoders import CnnEncoder, CnnHighwayEncoder, BagOfEmbeddingsEncoder, \
    PytorchSeq2VecWrapper, GruSeq2VecEncoder, \
    LstmSeq2VecEncoder, \
    StackedBidirectionalLstmSeq2VecEncoder, RnnSeq2VecEncoder, AugmentedLstmSeq2VecEncoder, \
    StackedAlternatingLstmSeq2VecEncoder , ClsPooler , BertPooler
from allennlp_models.lm.modules.language_model_heads.bert import  BertForMaskedLM


##############################################################################