
allennlp train model_config/action_desc_lstm_classifier.jsonnet -f -s runs --include-package self_allennlp
allennlp train model_config/bert_classifier.jsonnet -f -s runs --include-package self_allennlp

allennlp evaluate runs/model.tar.gz /home/jmz/Workstation/lb/Projects/data/action_desc/test.tsv --include-package self_allennlp


allennlp predict runs/model.tar.gz data/movie_review/test.jsonl --predictor sentence_classifier --include-package self_allennlp

# /home/liubin/tutorials/data/action_desc/test.tsv
# /home/jmz/Workstation/lb/Projects/data/action_desc



