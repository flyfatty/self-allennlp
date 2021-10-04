# @Time : 2021/1/1 11:23
# @Author : LiuBin
# @File : mt.py
# @Description : 
# @Software: PyCharm
from itertools import islice
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import LettersDigitsTokenizer, CharacterTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp_models.generation.dataset_readers import Seq2SeqDatasetReader
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.attention import DotProductAttention
from allennlp_models.generation import SimpleSeq2Seq
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp_models.generation.predictors import Seq2SeqPredictor
from allennlp.data.dataloader import PyTorchDataLoader


def main():
    EN_EMBEDDING_DIM = 256
    ZH_EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    CUDA_DEVICE = 0
    reader = Seq2SeqDatasetReader(
        source_tokenizer=LettersDigitsTokenizer(),
        target_tokenizer=CharacterTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    train_dataset = reader.read('data/tatoeba/tatoeba.eng_cmn.train.tsv')
    validation_dataset = reader.read('data/tatoeba/tatoeba.eng_cmn.dev.tsv')

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})
    train_dataset.index_with(vocab)
    validation_dataset.index_with(vocab)
    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)
    # encoder = PytorchSeq2SeqWrapper(
    #     torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128,
                                          feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()

    max_decoding_steps = 20  # TODO: make this variable
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8,
                          use_bleu=True)

    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]

    optimizer = AdamOptimizer(parameters)

    train_loader = PyTorchDataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_loader = PyTorchDataLoader(validation_dataset, batch_size=8, shuffle=False)


    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     validation_data_loader=dev_loader,
                                     num_epochs=5,
                                     cuda_device=CUDA_DEVICE)

    for i in range(50):
        print('Epoch: {}'.format(i))
        trainer.train()

        predictor = Seq2SeqPredictor(model, reader)

        for instance in islice(validation_dataset, 10):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])
