local root_path = "/data";
local bert_model = root_path + "/Pretrained_Model/bert-base-uncased";
local max_length = 512;
{
    "train_data_path": root_path + "/movie_review/train.tsv",
    "validation_data_path": root_path + "/movie_review/valid.tsv",

    "dataset_reader" : {
        "type": "cls_tsv_dataset_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": max_length
        },
        "token_indexer": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
                "max_length": max_length
            }
        }
    },
    "model": {
        "type": "basic_classifier_f",
        "text_field_embedder": {
            "type":"basic",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "seq2vec_encoder": {
            "type" : "bert_pooler",
            "pretrained_model" : bert_model,
        }
    },
    "data_loader": {
        "batch_size": 8,
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
