# @Time : 2021/4/5 1:42
# @Author : LiuBin
# @File : __init__.py.py
# @Description : 
# @Software: PyCharm

####################################################### sklearn #######################################################
# https://scikit-learn.org/stable/modules/classes.html?highlight=metric#module-sklearn.metrics

# > Metrics of Classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, \
    precision_recall_fscore_support, classification_report, confusion_matrix, f1_score, \
    fbeta_score

# > Metrics of Classification [plot]
from sklearn.metrics import precision_recall_curve, auc

# > Metrics of Classification [loss]
from sklearn.metrics import log_loss, hamming_loss, hinge_loss, zero_one_loss
####################################################### pytorch #######################################################


# tn , fp , fn , tp =  # [TN FP] [ FN TP ]