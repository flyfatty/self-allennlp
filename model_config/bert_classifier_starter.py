import tempfile
from typing import Dict, Iterable, List, Tuple
import os
import torch
import shutil

import allennlp
from allennlp.data import PyTorchDataLoader, DatasetReader, Instance, Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate
from allennlp.modules.seq2vec_encoders import BertPooler

from config import ConfigManager
from self_allennlp.data import ClsTsvDataSetReader, PretrainedTransformerTokenizer
from self_allennlp.models import BasicClassifierF, BertClassifier
from self_allennlp.predictors import SentenceClassifierPredictor

config = ConfigManager()
MODE = 'train'

DATA_PATH = os.path.join(config.DATA_PATH, "movie_review")
serialization_dir = os.path.join(config.DATA_PATH, "runs")
bert_model = os.path.join(config.DATA_PATH, "Pretrained_Model/bert-base-uncased")


# 构建 DatasetReader
def build_dataset_reader() -> DatasetReader:
    toeknizer = PretrainedTransformerTokenizer(bert_model, max_length=512)
    indexer = {'tokens': PretrainedTransformerIndexer(bert_model,max_length=512)}
    return ClsTsvDataSetReader(tokenizer=toeknizer, token_indexer=indexer)


# 加载数据
def read_data(
        reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read(os.path.join(DATA_PATH, "train.tsv"))
    validation_data = reader.read(os.path.join(DATA_PATH, "valid.tsv"))
    return training_data, validation_data


# 生成词表
def build_vocab(instances: Iterable[Instance] = None) -> Vocabulary:
    print("Building the vocabulary")
    vocab_path = os.path.join(DATA_PATH, "vocab")
    if os.path.exists(vocab_path):
        vocab = Vocabulary.from_files(vocab_path)
    else:
        vocab = Vocabulary.from_instances(instances)
        vocab.save_to_files(vocab_path)
    return vocab


# 构建DataLoader
def build_data_loaders(
        train_data: torch.utils.data.Dataset,
        dev_data: torch.utils.data.Dataset
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    batch_size = 8
    train_loader = PyTorchDataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = PyTorchDataLoader(dev_data, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader


# 构造模型
def build_model(vocab: Vocabulary) -> Model:
    bert_embedder = PretrainedTransformerEmbedder(bert_model)
    embedder = BasicTextFieldEmbedder({"tokens": bert_embedder})
    print("Building the model")
    model = BertClassifier(vocab, embedder=embedder)
    return model


# 构造训练器
def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: PyTorchDataLoader,
        dev_loader: PyTorchDataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    # optimizer = AdamW(parameters, lr=2e-5, eps=1e-8)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=1,
        optimizer=optimizer,
    )
    return trainer


def run_training_loop():
    if os.path.exists(serialization_dir):
        shutil.rmtree(os.path.join(serialization_dir))

    dataset_reader = build_dataset_reader()
    train_data, dev_data = read_data(dataset_reader)
    vocab = build_vocab(train_data + dev_data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    model = build_model(vocab)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    trainer = build_trainer(
        model,
        serialization_dir,
        train_loader,
        dev_loader
    )

    print("Starting training")
    trainer.train()
    print("Finished training")
    return model, dataset_reader


if __name__ == '__main__':

    if MODE == 'train':
        model, dataset_reader = run_training_loop()
        test_data = dataset_reader.read(os.path.join(DATA_PATH, 'test.tsv'))
        test_data.index_with(model.vocab)
        data_loader = PyTorchDataLoader(test_data, batch_size=8)
        results = evaluate(model, data_loader)
        print(results)
    else:
        vocab = build_vocab()
        dataset_reader = build_dataset_reader()
        model = build_model(vocab)
        model.load_state_dict(torch.load(open(os.path.join(serialization_dir, 'best.th'), 'rb')))
        if MODE == 'test':
            test_data = dataset_reader.read(os.path.join(DATA_PATH, 'test.tsv'))
            test_data.index_with(model.vocab)
            data_loader = PyTorchDataLoader(test_data, batch_size=8)
            results = evaluate(model, data_loader)
            print(results)
        elif MODE == 'predict':
            predictor = SentenceClassifierPredictor(model, dataset_reader)

            output = predictor.predict('A good movie!')
            print([(vocab.get_token_from_index(label_id, 'labels'), prob)
                   for label_id, prob in enumerate(output['probs'])])
            output = predictor.predict('This was a monstrous waste of time.')
            print([(vocab.get_token_from_index(label_id, 'labels'), prob)
                   for label_id, prob in enumerate(output['probs'])])
