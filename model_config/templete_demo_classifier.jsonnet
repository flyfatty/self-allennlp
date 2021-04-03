local lr = 0.001;
local weight_decay = 0.01;
local warmup_steps = 10;
local model_size = 128;

{
    root_path :: "/home/liubin/data",

    "train_data_path": $.root_path + "/movie_review/train.tsv",
    "validation_data_path": $.root_path + "/movie_review/valid.tsv",

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
             "type": "huggingface_adamw",
             "lr": lr,
             "weight_decay": weight_decay
         },
        "num_epochs" : 1,
        "learning_rate_scheduler" : {
            "type": "noam",
            "model_size": model_size,
            "warmup_steps": warmup_steps
        }
    }
}