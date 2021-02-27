
{
    root_path :: "data",

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
            "type":"boe"  ,
            "embedding_dim" : 128
        }
    },

    "data_loader":{
        "batch_size": 32,
        "shuffle":true
    },

    "trainer":{
        "optimizer" :"adam",
        "num_epochs" : 3
    }
}