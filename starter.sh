


# allennlp train model_config/templete_demo_classifier.jsonnet -f -s /home/liubin/data/movie_review/runs --include-package self_allennlp

# allennlp train model_config/action_desc_lstm_classifier.jsonnet -f -s runs --include-package self_allennlp

# allennlp train model_config/bert_classifier.jsonnet -f -s /home/liubin/data/movie_review/runs --include-package self_allennlp

# allennlp train model_config/templete_lstm_crf_tagger.jsonnet -f -s /home/liubin/data/movie_review/runs --include-package self_allennlp

# allennlp evaluate runs/model.tar.gz /home/jmz/Workstation/lb/Projects/data/action_desc/test.tsv --include-package self_allennlp

# allennlp predict runs/model.tar.gz data/movie_review/test.jsonl --predictor sentence_classifier --include-package self_allennlp

############################################### Optuna #############################################################
allennlp tune \
    model_config/templete_demo_classifier_with_optuna.jsonnet \
    model_config/hparams.json \
    --include-package self_allennlp \
    --serialization-dir result \
    --study-name demo \
    --optuna-param-path model_config/optuna.json \
    --storage XXXX \
    --skip-if-exists

# /home/liubin/tutorials/data/action_desc/test.tsv
# /home/jmz/Workstation/lb/Projects/data/action_desc


