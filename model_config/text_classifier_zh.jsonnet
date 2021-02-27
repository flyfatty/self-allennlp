
{

    root_path :: "/home/liubin/tutorials/pytorch/self-allennlp/data",

    "train_data_path": $.root_path + "/cnews/train.tsv",
    "validation_data_path": $.root_path + "/cnews/dev.tsv",

    "dataset_reader":{
        "type": "cls_tsv_dataset_reader",
        "tokenizer":"jieba",
        "label_first":true
    },

    "model":{
        "type": "basic_classifier",
        "text_field_embedder":{
            "token_embedders":{
                "tokens":{
                    "type": "embedding",
                    "embedding_dim": 128
                }
            }
        },
        "seq2vec_encoder":{
            "type":"rnn"  ,
            "input_size" : 128 ,
            "hidden_size" : 128
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