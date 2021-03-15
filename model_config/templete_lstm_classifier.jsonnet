
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
                    "embedding_dim":128
                }
            }
        },
        "seq2vec_encoder":{
            "type":"lstm"  ,
            "hidden_size" : 128,
            "input_size": 128,
            "num_layers": 1,
            "bidirectional": false,
            "layer_dropout_probability": 0
        }
    },

    "data_loader":{
        "batch_size": 16,
        "shuffle": true
    },

    "trainer":{
        "optimizer" :"adam",
        "num_epochs" : 3
    }
}