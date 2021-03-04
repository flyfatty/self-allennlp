
{
    root_path :: "/home/liubin/data",

    "train_data_path": $.root_path + "/people_daily/train.tsv",
    "validation_data_path": $.root_path + "/people_daily/valid.tsv",

    "dataset_reader":{
        "type": "tag_tsv_dataset_reader",
        "tokenizer":{
            "type":"whitespace"
        },
        "token_indexer":{
            "tokens":{
                "type":"single_id"
            }
        },
//        "limit" : 100
    },
    "model":{
        "type": "crf_tagger",
        "text_field_embedder":{
            "type":"basic",
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    "embedding_dim": 200,
                    "pretrained_file" : $.root_path + "/Embedding/Tencent_AILab_ChineseEmbedding.txt"
                }
            }
        },
        "encoder":{
            "type":"lstm" ,
            "hidden_size" : 200 ,
            "input_size": 200 ,
            "num_layers": 1 ,
            "bidirectional": true
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