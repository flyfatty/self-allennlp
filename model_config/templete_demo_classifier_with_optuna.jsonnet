local lr = std.parseJson(std.extVar('lr'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local warmup_steps = std.parseInt(std.extVar('warmup_steps'));
local model_size = std.parseJson(std.extVar('model_size'));

{
    root_path :: "data/movie_review",

    "train_data_path": $.root_path + "/train.tsv",
    "validation_data_path": $.root_path + "/valid.tsv",

    "dataset_reader":{
        "type": "cls_tsv_dataset_reader",
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
        "type": "basic_classifier",
        "text_field_embedder":{
            "type":"basic",
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    "embedding_dim":model_size
                }
            }
        },
        "seq2vec_encoder":{
            "type":"boe"  ,
            "embedding_dim" : model_size
        }
    },

    "data_loader":{
        "batch_size": 32,
        "shuffle": true
    },

    "trainer":{
        "type":"gradient_descent",
        "optimizer" :{
             "type": "sgd",
             "lr": lr,
             "momentum": 0.9,
             "nesterov": true,
             "weight_decay": weight_decay
         },
        "num_epochs" : 32,
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