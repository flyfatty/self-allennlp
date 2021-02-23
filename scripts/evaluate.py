import tempfile
from typing import Dict, Iterable, List, Tuple
import os
import torch

import allennlp
from allennlp.data import PyTorchDataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate
from allennlp.models import BasicClassifier
from self_allennlp.data import ClsTsvDataSetReader
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
from self_allennlp.data import ClsTsvDataSetReader, JiebaTokenizer
from self_allennlp.models import SimpleClassifier

DATA_PATH = "/home/liubin/tutorials/data/action_desc"
EMBEDDING_FILE = "/home/liubin/tutorials/data/action_desc/embedding.h5"
serialization_dir = "/home/liubin/tutorials/data/action_desc/runs"


def build_dataset_reader() -> DatasetReader:
    return ClsTsvDataSetReader(tokenizer=JiebaTokenizer())


def read_data(
        reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read(os.path.join(DATA_PATH, "train.tsv"))
    validation_data = reader.read(os.path.join(DATA_PATH, "valid.tsv"))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    vocab = Vocabulary.from_files(os.path.join(DATA_PATH, "vocab"))
    return vocab


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    # vocab_size = vocab.get_vocab_size("tokens")
    embedding = Embedding(embedding_dim=200, vocab=vocab, pretrained_file=EMBEDDING_FILE)
    # with h5py.File(os.path.join(DATA_PATH, "embedding.h5"), 'w') as f:
    #     f.create_dataset('embedding', data=embedding.weight.data)
    embedder = BasicTextFieldEmbedder(
        {"tokens": embedding}
    )

    encoder = LstmSeq2VecEncoder(input_size=200, hidden_size=256)
    # encoder = BagOfEmbeddingsEncoder(embedding_dim=200)
    return SimpleClassifier(vocab, embedder, encoder)


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
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer


# The other `build_*` methods are things we've seen before, so they are
# in the setup section above.
def build_data_loaders(
        train_data: torch.utils.data.Dataset,
        dev_data: torch.utils.data.Dataset,
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = PyTorchDataLoader(train_data, batch_size=8, shuffle=True)
    dev_loader = PyTorchDataLoader(dev_data, batch_size=8, shuffle=False)
    return train_loader, dev_loader


def run_training_loop():
    dataset_reader = build_dataset_reader()

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    return model, dataset_reader


# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
model, dataset_reader = run_training_loop()
model = model.load(serialization_dir=serialization_dir)
# Now we can evaluate the model on a new dataset_reader.
test_data = dataset_reader.read(os.path.join(DATA_PATH, 'test.tsv'))

test_data.index_with(model.vocab)
data_loader = PyTorchDataLoader(test_data, batch_size=8)

results = evaluate(model, data_loader)
print(results)
