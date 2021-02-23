
{
    root_path :: "/home/liubin/tutorials/pytorch/self-allennlp/data",

    "train_data_path": $.root_path + "/movie_review/train.tsv",
    "validation_data_path": $.root_path + "/movie_review/valid.tsv",

    "dataset_reader":{
        "type": "cls_tsv_dataset_reader",
        "tokenizer":{
            "type":"whitespace"
        },
        "token_indexer":{
            "index1":{
                "type":"single_id"
            }
        }
    },
    "model":{
        "type": "basic_classifier",
        "text_field_embedder":{
            "type":"basic",
            "token_embedders":{
                "index1":{
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