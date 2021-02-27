
{
    root_path :: "/home/liubin/tutorials/data/action_desc",
    pretrained_file :: $.root_path + "/embedding.h5",

    "train_data_path": $.root_path + "/train.tsv",
    "validation_data_path": $.root_path + "/valid.tsv",

    "dataset_reader":{
        "type": "cls_tsv_dataset_reader",
        "tokenizer":{
            "type":"jieba"
        },
        "token_indexer":{
            "tokens":{
                "type":"single_id"
            }
        },
//        "limit":10
    },
    "model":{
        "type": "basic_classifier_f",
        "text_field_embedder":{
            "type":"basic",
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    "embedding_dim":200,
                    "pretrained_file":$.pretrained_file
                }
            }
        },
        "seq2vec_encoder":{
            "type":"stacked_bidirectional_lstm"  ,
            "input_size" : 200,
            "hidden_size" : 256,
            "num_layers":2,
            "layer_dropout_probability":0.2
        }
    },

    "data_loader":{
        "batch_size": 32,
        "shuffle":true
    },

    "trainer":{
        "optimizer" :"adam",
        "num_epochs" : 5
    }
}