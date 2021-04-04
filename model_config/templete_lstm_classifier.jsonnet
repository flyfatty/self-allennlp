local lr = 1e-3;
local weight_decay = 1e-3;
local warmup_steps = 300;
local model_size = 128;
local num_layers = 1;
local dropout = 0;
local bidirectional = false;
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
        }
    }
}