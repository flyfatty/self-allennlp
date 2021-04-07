local cuda_device = 0;
local num_epochs = 8;
local seed = 42;
local batch_size = std.parseInt(std.extVar('batch_size'));
local lr = std.parseJson(std.extVar('lr'));
local warmup_steps = std.parseInt(std.extVar('warmup_steps'));
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local model_size = std.parseJson(std.extVar('model_size'));
local num_layers = std.parseInt(std.extVar('num_layers'));
local dropout = std.parseJson(std.extVar('dropout'));
local bidirectional = std.parseJson(std.extVar('bidirectional'));

{
    root_path :: "data/news-topic",

    "train_data_path": $.root_path + "/train.tsv",
    "validation_data_path": $.root_path + "/valid.tsv",

    "dataset_reader":{
        "type": "cls_tsv_dataset_reader",
        "label_first": true,
        "tokenizer":{
            "type":"whitespace"
        },
        "token_indexer":{
            "tokens":{
                "type":"single_id"
            }
        }
    },
    "model":{
        "type": "basic_classifier_f",
        "text_field_embedder":{
            "type":"basic",
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    "embedding_dim":embedding_dim
                }
            }
        },
        "seq2vec_encoder":{
            "type":"lstm"  ,
            "hidden_size" : model_size,
            "input_size": embedding_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout
        }
    },

    "data_loader":{
        "batch_size": batch_size,
        "shuffle": true
    },

    "trainer":{
        "type" : "gradient_descent",
        "cuda_device": cuda_device,
        "optimizer" :{
             "type": "adam",
             "lr": lr
        },
        "num_epochs" : num_epochs,
        "learning_rate_scheduler" : {
            "type": "noam",
            "model_size": model_size,
            "warmup_steps": warmup_steps
        },
        "epoch_callbacks": [
            {
              type: "optuna_pruner",
            }
       ]
    }
}