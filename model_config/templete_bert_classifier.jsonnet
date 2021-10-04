local bert_model = "data/Pretrained_Model/bert-base-uncased";
local max_length = 512;
local cuda_device = 0;
local num_epochs = 4;
local batch_size = 16;
local lr = 1e-5;


{
    root_path :: "data/news-topic",

    "train_data_path": $.root_path + "/train.tsv",
    "validation_data_path": $.root_path + "/valid.tsv",

    "dataset_reader" : {
        "type": "cls_tsv_dataset_reader",
        "label_first": true,
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
            "requires_grad": false
        }
    },
    "data_loader": {
        "batch_size": batch_size,
        "shuffle": true
    },
    "trainer": {
        "cuda_device": cuda_device,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": lr
        },
        "num_epochs": num_epochs
    }
}
