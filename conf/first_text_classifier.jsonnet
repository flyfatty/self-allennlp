{
    root_path :: "/home/liubin/tutorials/pytorch/self-allennlp/data",
    "dataset_reader" : {
        "type": "cls_tsv_dataset_reader",
        "token_indexer": {
            "index1": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": $.root_path + "/movie_review/train.tsv",
    "validation_data_path": $.root_path + "/movie_review/valid.tsv",
    "model": {
        "type": "simple_classifier", // Model名称
        "vocab": {
            "type":"from_files",
        },
        "embedder": {
            "token_embedders": {
                "index1": {
                    "type":"embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": { // Model参数
            "type": "bag_of_embeddings", // 内置 Embedder名称
            "embedding_dim": 10
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
