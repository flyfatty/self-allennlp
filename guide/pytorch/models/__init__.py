# @Time : 2021/3/24 0:41
# @Author : LiuBin
# @File : __init__.py.py
# @Description : 
# @Software: PyCharm

from transformers import AutoTokenizer, AutoModelForMaskedLM , RobertaTokenizer , RobertaModel,BertTokenizer, BertModel
#
tokenizer = BertTokenizer.from_pretrained("/home/liubin/data/Pretrained_Model/bert-base-uncased/")

model = BertModel.from_pretrained("/home/liubin/data/Pretrained_Model/bert-base-uncased/")
# tokenizer = BertTokenizer.from_pretrained('/home/liubin/data/Pretrained_Model/chinese-roberta-wwm-ext')
# model = BertModel.from_pretrained('/home/liubin/data/Pretrained_Model/chinese-roberta-wwm-ext')
text = "巴黎是[MASK]国的首都。"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)