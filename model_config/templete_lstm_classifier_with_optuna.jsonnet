local lr = std.parseJson(std.extVar('lr'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local warmup_steps = std.parseInt(std.extVar('warmup_steps'));
local model_size = std.parseJson(std.extVar('model_size'));
local num_layers = std.parseInt(std.extVar('num_layers'));
local dropout = std.parseJson(std.extVar('dropout'));
local bidirectional = std.parseJson(std.extVar('bidirectional'));

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
            "type":"lstm"  ,
            "hidden_size" : model_size,
            "input_size": model_size,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout
        }
    },

    "data_loader":{
        "batch_size": 16,
        "shuffle": true
    },

    "trainer":{
        "type" : "gradient_descent",
        "optimizer" :{
             "type": "adamw",
             "lr": lr,
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