# self-allennlp

### Guide
* 示例数据集
    * 电影评价-情感分析 movie_review


## 模型配置
#### 文本分类（CLS）
* BOE (Demo)
  * templete_demo_classifier.jsonnet
  * templete_demo_classifier_with_optuna.jsonnet
* LSTM
 * templete_lstm_classifier.jsonnet
* Bert
 * bert_classifier.jsonnet

#### 序列标注（POS）
* HMM / MEMMs   `学习p(x,y)`
* CRF      `sentence-level tag information 学习p(y|x)`
* Conv-CRF
* LSTM / BI-LSTM
* LSTM-CRF / BI-LSTM-CRF `less dependence on word embedding`