import tempfile
from typing import Dict, Iterable, List, Tuple
import os
import torch
import shutil

import allennlp

from allennlp.nn import Activation
from allennlp.data import PyTorchDataLoader, DatasetReader, Instance, Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.feedforward import FeedForward
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BagOfEmbeddingsEncoder
from self_allennlp.data import ClsAbsJsonlDataSetReader, JiebaTokenizer
from self_allennlp.models.abs_classifier import AbsClassifier
from self_allennlp.predictors import SentenceClassifierPredictor

MODE = 'train'
DATA_PATH = "/home/liubin/tutorials/data/academic-papers"
# DATA_PATH = "/home/liubin/tutorials/pytorch/self-allennlp/data/movie_review"
EMBEDDING_FILE = os.path.join(DATA_PATH, "embedding.h5")
serialization_dir = os.path.join(DATA_PATH, "runs")


# 构建 DatasetReader
def build_dataset_reader() -> DatasetReader:
    return ClsAbsJsonlDataSetReader(key_map={'title': 'title', 'abstract': 'paperAbstract', 'label': 'venue'})


# 加载数据
def read_data(
        reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read(os.path.join(DATA_PATH, "train.jsonl"))
    validation_data = reader.read(os.path.join(DATA_PATH, "valid.jsonl"))
    return training_data, validation_data


# 生成词表
def build_vocab(instances: Iterable[Instance] = None) -> Vocabulary:
    print("Building the vocabulary")
    if os.path.exists(os.path.join(DATA_PATH, "vocab")):
        vocab = Vocabulary.from_files(os.path.join(DATA_PATH, "vocab"))
    else:
        vocab = Vocabulary.from_instances(instances)
        vocab.save_to_files(os.path.join(DATA_PATH, "vocab"))
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
    print("Building the model")
    # vocab_size = vocab.get_vocab_size("tokens")
    # embedding = Embedding(embedding_dim=200, vocab=vocab, pretrained_file=EMBEDDING_FILE)
    embedding = Embedding(embedding_dim=100, vocab=vocab)
    # with h5py.File(os.path.join(DATA_PATH, "embedding.h5"), 'w') as f:
    #     f.create_dataset('embedding', data=embedding.weight.data)
    embedder = BasicTextFieldEmbedder(
        {"tokens": embedding}
    )

    title_encoder = LstmSeq2VecEncoder(input_size=100, bidirectional=True, hidden_size=100, dropout=0.2)
    abstrict_encoder = LstmSeq2VecEncoder(input_size=100, bidirectional=True, hidden_size=100, dropout=0.2)
    feedforward = FeedForward(abstrict_encoder.get_output_dim(),num_layers=2,hidden_dims=[200,3],activations=[Activation.by_name('relu')(),Activation.by_name('linear')()])
    model = AbsClassifier(vocab, embedder, title_encoder=title_encoder, abstract_encoder=abstrict_encoder,feedforward=feedforward)
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