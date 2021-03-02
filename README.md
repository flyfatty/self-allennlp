# self-allennlp
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
    
#### 序列标注（POS）
* HMM / MEMMs   `学习p(x,y)`
* CRF      `sentence-level tag information 学习p(y|x)`
* Conv-CRF
* LSTM / BI-LSTM
* LSTM-CRF / BI-LSTM-CRF `less dependence on word embedding`