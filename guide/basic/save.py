# @Time : 2020/12/18 9:38
# @Author : LiuBin
# @File : save.py
# @Description : 
# @Software: PyCharm

# Model config (specifications used to train the model)
# Model weights (trained parameters of the model)
# Vocabulary
import os
import json
import tempfile
import torch
import torch.nn.functional as F

from copy import deepcopy
from overrides import overrides
from typing import List, Dict, Iterable

from allennlp.predictors import Predictor
from allennlp.common import JsonDict
from allennlp.data import Instance, DatasetReader, DataLoader, Tokenizer, AllennlpDataset
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models import Model, archive_model, load_archive
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.training import Trainer
from allennlp.nn import util

from config import ConfigManager

config = ConfigManager()


@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                yield self.text_to_instance(text, sentiment)


@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = F.softmax(logits)
        # Shape: (1,)
        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


def run_config(config):
    params = Params(json.loads(config))
    params_copy = params.duplicate()

    if 'dataset_reader' in params:
        reader = DatasetReader.from_params(params.pop('dataset_reader'))
    else:
        raise RuntimeError('`dataset_reader` section is required')

    all_instances = []
    if 'train_data_path' in params:
        print('Reading the training data...')
        train_data = reader.read(params.pop('train_data_path'))
        all_instances.extend(train_data)
    else:
        raise RuntimeError('`train_data_path` section is required')

    validation_data = None
    if 'validation_data_path' in params:
        print('Reading the validation data...')
        validation_data = reader.read(params.pop('validation_data_path'))
        all_instances.extend(validation_data)

    print('Building the vocabulary...')
    vocab = Vocabulary.from_instances(all_instances)

    model = None

    train_data.index_with(vocab)
    validation_data.index_with(vocab)

    iterator = AllennlpDataset(all_instances)
    if 'model' not in params:
        # 'dataset' mode — just preview the (first 10) instances
        print('Showing the first 10 instances:')
        for inst in all_instances[:10]:
            print(inst)
    else:
        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        loader_params = deepcopy(params.pop("data_loader"))
        train_data_loader = DataLoader.from_params(dataset=train_data,
                                                   params=loader_params)
        dev_data_loader = DataLoader.from_params(dataset=validation_data,
                                                 params=loader_params)

        iterator.index_with(vocab)

        # set up a temporary, empty directory for serialization
        with tempfile.TemporaryDirectory() as serialization_dir:
            trainer = Trainer.from_params(
                model=model,
                serialization_dir=serialization_dir,
                data_loader=train_data_loader,
                validation_data_loader=dev_data_loader,
                params=params.pop('trainer'))
            trainer.train()

    return {
        'params': params_copy,
        'dataset_reader': reader,
        'vocab': vocab,
        'iterator': iterator,
        'model': model
    }


CONFIG = """
{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "../../data/movie_review/train.tsv",
    "validation_data_path": "../../data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "data_loader": {
        "batch_size": 8
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
"""


def make_predictions(model: Model, dataset_reader: DatasetReader) \
        -> List[Dict[str, float]]:
    """Make predictions using the given model and dataset reader."""
    predictions = []
    predictor = SentenceClassifierPredictor(model, dataset_reader)
    output = predictor.predict('A good movie!')
    predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
                        for label_id, prob in enumerate(output['probs'])})
    output = predictor.predict('This was a monstrous waste of time.')
    predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
                        for label_id, prob in enumerate(output['probs'])})
    return predictions


# Because we can't use bash to run allennlp commands, and so that we can more
# easily pull out some pieces to show you how this works, we wrote a simple method
# that runs a training loop from a configuration file. You can see it in the Setup
# section above.
components = run_config(CONFIG)
params = components['params']
dataset_reader = components['dataset_reader']
vocab = components['vocab']
model = components['model']

original_preds = make_predictions(model, dataset_reader)

# Save the model
serialization_dir = 'model'
config_file = os.path.join(serialization_dir, 'config.json')
vocabulary_dir = os.path.join(serialization_dir, 'vocabulary')
weights_file = os.path.join(serialization_dir, 'weights.th')

os.makedirs(serialization_dir, exist_ok=True)
params.to_file(config_file)
vocab.save_to_files(vocabulary_dir)
torch.save(model.state_dict(), weights_file)

# Load the model
loaded_params = Params.from_file(config_file)
loaded_model = Model.load(loaded_params, serialization_dir, weights_file)
loaded_vocab = loaded_model.vocab  # Vocabulary is loaded in Model.load()

# Make sure the predictions are the same
loaded_preds = make_predictions(loaded_model, dataset_reader)
assert original_preds == loaded_preds
print('predictions matched')

# Create an archive file
archive_model(serialization_dir, weights='weights.th')

# Unarchive from the file
archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'))

# Make sure the predictions are the same
archived_preds = make_predictions(archive.model, dataset_reader)
assert original_preds == archived_preds
print('predictions matched')
