# @Time : 2021/4/1 13:56
# @Author : LiuBin
# @File : tfidf_optuna.py
# @Description : 
# @Software: PyCharm
import os
import pandas as pd
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

from config import ConfigManager

config = ConfigManager()
# 加载 data
train_data_path = os.path.join(config.DATA_PATH, "movie_review", 'train.tsv')
test_data_path = os.path.join(config.DATA_PATH, "movie_review", 'train.tsv')
df = pd.read_table(train_data_path, names=['text', 'label'])
train_X, valid_X, train_y, valid_y = train_test_split(df['text'], df['label'])
df_test = pd.read_table(test_data_path, names=['text', 'label'])
# 构造 pipeline
pipeline = Pipeline([
    ('vec', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('cls', SGDClassifier(loss='log')),
])

# 设置 param
parameters = {
    'vec__binary': True,
    'vec__min_df': 1,
    'vec__max_df': 1.0,
    'vec__ngram_range': (1, 1),
    'tfidf__norm': 'l2',
    'cls__alpha': 0.00013429366000393444
}

parameters_cv = {
    'vec__binary': (False, True,),
    'vec__min_df': (1, 2,),
    'vec__max_df': (0.5, 1.0),
    'vec__ngram_range': ((1, 1), (1, 2)),
}

# single
pipeline.set_params(**parameters).fit(train_X, train_y)
score = pipeline.score(valid_X, valid_y)
test_score = pipeline.score(df_test['text'], df_test['label'])

print(score, test_score)

from sklearn.metrics import precision_recall_curve

presiton , recall , thresholds = precision_recall_curve(df_test['label'].values, pipeline.predict_proba(df_test['text'])[:, 0], pos_label='pos')
# sns.lineplot(x=recall,y=presiton)
# plt.show()
# exit(0)
# CV

grid_search = GridSearchCV(pipeline, parameters_cv, n_jobs=-1, verbose=1).fit(train_X, train_y)
print(grid_search.best_params_)
score_cv = grid_search.score(valid_X, valid_y)
print(score_cv)


# Optuna
def objective(trial: optuna.Trial) -> float:
    parameters_op = {
        'vec__binary': trial.suggest_categorical("binary", [False, True]),
        'vec__min_df': trial.suggest_int("min_df", 1, 1),
        'vec__max_df': trial.suggest_float("max_df", 1.0, 1.0),
        'vec__ngram_range': trial.suggest_categorical("ngram_range", [(1, 1), (1, 2)]),
        'tfidf__norm': trial.suggest_categorical("norm", ['l1', 'l2', None]),
        'cls__alpha': trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    }
    pipeline.set_params(**parameters_op).fit(train_X, train_y)

    return pipeline.score(valid_X, valid_y)


study = optuna.create_study(study_name="demo2", direction="maximize", load_if_exists=True,
                            storage="postgresql://liubin:ize4sg@localhost/logs")
# study = optuna.load_study(study_name="demo", storage="postgresql://liubin:ize4sg@localhost/logs")
study.optimize(objective, n_trials=100)
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
