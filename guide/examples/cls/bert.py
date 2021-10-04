# @Time : 2021/4/6 13:54
# @Author : LiuBin
# @File : bert.py
# @Description : 
# @Software: PyCharm

from transformers import AutoTokenizer, AutoModelForMaskedLM , BertTokenizer , BertModel
AutoTokenizer.from_pretrained("bert-base-uncased")


