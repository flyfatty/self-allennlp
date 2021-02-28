local bert_model = "bert-base-uncased";
local max_length = 512;
{
    "train_data_path": "data/movie_review/train.tsv",
    "validation_data_path": "data/movie_review/valid.tsv",

    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": max_length
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
                "max_length": max_length
            }
        }
    },

    "model": {
        "type": "bert_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        }
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 5
    }
}
