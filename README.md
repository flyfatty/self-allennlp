# self-allennlp

### Guide
* 示例任务
    * 新闻主题分类 news-topic
* 示例模型
    * BOW
    * TFIDF
    
###### Optuna

### 1、训练阶段

### 2、评价阶段
**Model**里建立Metrics，使用测试集进行评价
内置Metrics|说明
:---------:|--
CategoricalAccuracy| 
### 3、预测阶段

【训练】train   使用训练集和验证集训练模型
【评价】evaluate   使用测试集评价模型    Metrics



## 模型配置
#### 文本分类（CLS）
* BOE
 * demo_classifier.jsonnet
* LSTM
 * lstm_classifier.jsonnet
* Bert
 * bert_classifier.jsonnet

#### 序列标注（POS）
* HMM / MEMMs   `学习p(x,y)`
* CRF      `sentence-level tag information 学习p(y|x)`
* Conv-CRF
* LSTM / BI-LSTM
* LSTM-CRF / BI-LSTM-CRF `less dependence on word embedding`