# @Time : 2021/4/1 13:56
# @Author : LiuBin
# @File : tfidf_optuna.py
# @Description : 
# @Software: PyCharm
import os
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

from config import ConfigManager

config = ConfigManager()
# 加载 data
train_data_path = os.path.join(config.DATA_PATH, "movie_review", 'train.tsv')
df = pd.read_table(train_data_path, names=['text', 'label'])
train_X, valid_X, train_y, valid_y = train_test_split(df['text'], df['label'])

# 构造 pipeline
pipeline = Pipeline([
    ('vec', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('cls', SGDClassifier()),
])

# 设置 param
parameters = {
    'vec__binary': True,
    'vec__min_df': 1,
    'vec__max_df': 1.0,
    'vec__ngram_range': (1, 1),
}

parameters_cv = {
    'vec__binary': (False, True,),
    'vec__min_df': (1, 2,),
    'vec__max_df': (0.5, 1.0),
    'vec__ngram_range': ((1, 1), (1, 2)),
}


# 开始 train
pipeline.set_params(**parameters).fit(train_X, train_y)
grid_search = GridSearchCV(pipeline, parameters_cv, n_jobs=-1, verbose=1).fit(train_X, train_y)
print(grid_search.best_params_)
# 开始 eval
score = pipeline.score(valid_X, valid_y)
score_cv = grid_search.score(valid_X, valid_y)
print(score, score_cv)

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


study = optuna.create_study(study_name="demo", direction="maximize", load_if_exists=True,
                            storage="postgresql://liubin:ize4sg@localhost/logs")
# study = optuna.load_study(study_name="demo", storage="postgresql://liubin:ize4sg@localhost/logs")
study.optimize(objective, n_trials=100)
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
