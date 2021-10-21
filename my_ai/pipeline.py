import os
import math
from pprint import pprint
import logging

import torch
import numpy as np
import pickle as pkl

import my_ai.utility

UKN, PAD = '<ukn>', '<pad>'


# region Config
class ConfigFactory:
    @staticmethod
    def get_config(model_name, files_path, params=None, other_params=None):
        if model_name is None:
            raise Exception("model_name is None!")
        if files_path is None:
            raise Exception("files_path is None!")

        if model_name == "TextCNN":
            default_params = ConfigFactory.get_text_classify_params(model_name, files_path)
            default_other_params = {
                'filter_sizes': (2, 3, 4),
                'num_filters': 256
            }
            params = ConfigFactory.fill_params(params, default_params)
            other_params = ConfigFactory.fill_params(other_params, default_other_params)
            config = TextClassifyConfig(params, other_params)
        else:
            raise Exception("unrecognized model_name!")
        return config

    @staticmethod
    def fill_params(params, default_params):
        if params is None:
            params = default_params
        else:
            for key in default_params:
                if params.get(key) is None:
                    params[key] = default_params[key]
        return params

    @staticmethod
    def get_text_classify_params(model_name, files_path):
        params = {
            'model_name': model_name,
            'files_path': files_path,
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'is_char_segment': 1,
            'min_freq': 2,
            'is_revocab': 1,
            'is_retrim_embedding': 0,
            'is_pretrained': 1,
            'embedding_length': 300,
            'dropout': 0.5,
            'num_epochs': 20,
            'batch_size': 128,
            'text_length': 30,
            'learning_rate': 1e-3,
            'start_expire_after': 1,  # after how many epochs begin counting expire
            'expire_batches': 1000,  # early drop after {expire_batches} batches without improvement
        }
        return params


class TextClassifyConfig:
    def __init__(self, params: dict, other_params: dict):
        # files path
        self.files_path = params['files_path']
        self.train_path = params['files_path'] + '/train.txt'
        self.dev_path = params['files_path'] + '/dev.txt'
        self.test_path = params['files_path'] + '/test.txt'
        self.vocab_path = params['files_path'] + '/vocab.pkl'
        self.save_path = params['files_path'] + '/saved_dict.ckpt'
        self.trimmed_embed_path = params['files_path'] + '/trimmed_embedding.npz'
        self.original_embed_path = params['files_path'] + '/original_embedding'
        self.log_path = params['files_path'] + '/log'

        # basic info
        self.model_name = params['model_name']
        self.device = params['device']
        self.is_char_segment = params['is_char_segment']
        self.min_freq = params['min_freq']
        self.is_revocab = params['is_revocab']
        self.is_retrim_embedding = params['is_retrim_embedding']
        self.is_pretrained = params['is_pretrained']
        self.embedding_length = params['embedding_length']
        self.dropout = params['dropout']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.text_length = params['text_length']
        self.learning_rate = params['learning_rate']
        self.start_expire_after = params['start_expire_after']
        self.expire_batches = params['expire_batches']

        self.class_list = None
        self.num_classes = None
        self.tokenizer: Tokenizer = None
        self.vocab: Vocab = None
        self.embedding: Embedding = None
        self.train_dataloader: Dataloader = None
        self.dev_dataloader: Dataloader = None
        self.test_dataloader: Dataloader = None
        # other_params
        self.other_params = other_params

    def show(self):
        pprint(vars(self))


# endregion

# region Preprocessor
class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(config):
        if config.model_name == 'TextCNN':
            preprocessor = TextClassifyPreprocessor(config)
        else:
            raise Exception("unrecognized model_name!")
        return preprocessor


class TextClassifyPreprocessor:
    def __init__(self, config: TextClassifyConfig):
        self.config = config

    def preprocess(self):
        # preprocess config
        self.config.class_list = [x.strip() for x in open(
            self.config.files_path + '/class.txt', encoding='utf-8').readlines()]
        self.config.num_classes = len(self.config.class_list)
        # tokenizer
        logging.info('--Begin  tokenizer.')
        logging.debug(f'\r\n\
           config.is_char_segment:{self.config.is_char_segment}\
        ')
        self.config.tokenizer = self._get_tokenizer()
        logging.info('--Finished  tokenizer.')
        # vocab
        logging.info('--Begin  vocab.')
        self.config.vocab = self._get_vocab()
        if self.config.is_revocab == 1:
            logging.info('Start building vocab.')
            self.config.vocab.build_vocab_of_sentences()
        else:
            logging.info('Start reloading  vocab.')
            self.config.vocab.load_vocab()
        logging.info('--Finished vocab.')
        # embedding
        logging.info('--Begin  embedding.')
        logging.debug(f'\r\n\
           config.trimmed_embed_path:{self.config.trimmed_embed_path}\r\n\
           config.original_embed_path:{self.config.original_embed_path}\
        ')
        self.config.embedding = self._get_embedding()
        if self.config.is_retrim_embedding == 1:
            logging.info('Start building  embedding.')
            self.config.embedding.build_trimmed()
        else:
            logging.info('Start reloading  embedding.')
            self.config.embedding.load_trimmed()
        logging.debug(f'\r\n\
           left index with its token:{self.config.embedding.get_left_index_token()}\
        ')
        logging.info('--Finished  embedding.')
        # dataloader
        logging.info('--Begin  dataloader.')
        logging.debug(f'\r\n\
           config.batch_size:{self.config.batch_size}\r\n\
           config.text_length:{self.config.text_length}\
        ')
        self.config.train_dataloader, self.config.dev_dataloader, self.config.test_dataloader = self._get_dataloader()
        logging.info('--Finished  dataloader.')
        return self

    def _get_dataloader(self):
        train_dataset = TextClassifyDataset(self.config.train_path, self.config.text_length, self.config.vocab,
                                            self.config.tokenizer)
        train_dataloader = Dataloader(train_dataset, self.config.batch_size)
        dev_dataset = TextClassifyDataset(self.config.dev_path, self.config.text_length, self.config.vocab,
                                          self.config.tokenizer)
        dev_dataloader = Dataloader(dev_dataset, self.config.batch_size)
        test_dataset = TextClassifyDataset(self.config.test_path, self.config.text_length, self.config.vocab,
                                           self.config.tokenizer)
        test_dataloader = Dataloader(test_dataset, self.config.batch_size)
        return train_dataloader, dev_dataloader, test_dataloader

    def _get_tokenizer(self):
        tokenizer = Tokenizer(self.config.is_char_segment)
        return tokenizer

    def _get_vocab(self):
        vocab = Vocab(self.config.train_path, self.config.vocab_path, self.config.tokenizer, self.config.min_freq)
        return vocab

    def _get_embedding(self):
        embedding = Embedding(self.config.trimmed_embed_path, self.config.original_embed_path, self.config.vocab)
        return embedding


class Tokenizer:
    def __init__(self, is_char_segment):
        self.is_char_segment = is_char_segment

    def tokenize(self, text):
        if self.is_char_segment:
            return self.tokenize_by_char(text)
        else:
            return self.tokenize_by_word(text)

    def tokenize_by_char(self, text):
        result = [char for char in text]
        return result

    def tokenize_by_word(self, text):
        result = text.split(' ')
        return result


class Vocab:
    def __init__(self, train_path, vocab_path, tokenizer: Tokenizer, min_freq=2):
        self.train_path = train_path
        self.vocab_path = vocab_path
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        self.idx_to_token, self.token_to_idx = [PAD, UKN], {PAD: 0, UKN: 1}

    def build_vocab_of_sentences(self):
        vocab_dic = {}
        separator = '\t'

        with open(self.train_path, 'r', encoding='UTF-8') as f:
            for line in f:
                sentence = line.strip().split(separator)[0]
                if sentence == '':
                    continue
                for word in self.tokenizer.tokenize(sentence):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1

        i = 2
        for key in vocab_dic:
            if key == PAD or key == UKN:
                continue
            if vocab_dic[key] >= self.min_freq:
                self.idx_to_token.append(key)
                self.token_to_idx[key] = i
                i = i + 1

        pkl.dump(self.idx_to_token, open(self.vocab_path, 'wb'))

    def load_vocab(self):
        self.idx_to_token = pkl.load(open(self.vocab_path, 'rb'))
        self.token_to_idx = {}
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def get_len(self):
        return len(self.idx_to_token)

    def __len__(self):
        return len(self.idx_to_token)

    def to_token(self, index):
        return self.idx_to_token[index]

    def to_index(self, token):
        if token == PAD or token == UKN:
            return 1
        return self.token_to_idx.get(token, 1)


class Embedding:
    def __init__(self, trimmed_path: str, original_path: str = '', vocab: Vocab = None):
        self.original_path = original_path
        self.trimmed_path = trimmed_path
        self.vocab = vocab
        self.representation = None
        self.len = None
        self.left_index = []
        # [PAD, UKN]
        self.special_index = [0, 1]

    def build_trimmed(self):
        original_has_header = True
        # embedding_len
        with open(self.original_path, 'r', encoding='UTF-8') as f:
            if original_has_header:
                f.readline()
            elems = f.readline().strip().split()
            self.len = len(elems[1:])
        # representation of embedding
        self.representation = np.zeros((self.vocab.get_len(), self.len))

        with open(self.original_path, 'r', encoding='UTF-8') as f:
            if original_has_header:
                f.readline()
            for line in f:
                elems = line.strip().split()
                vocab_index = self.vocab.to_index(elems[0])
                if vocab_index == -1:
                    continue
                # if original embedding has PAD or UKN ,just abandon it bc it may has diff meaning
                if elems[0] == PAD or elems[0] == UKN:
                    continue
                self.representation[vocab_index, 0:] = elems[1:]

        zero = np.zeros(self.len)
        for vocab_index in range(self.vocab.get_len()):
            if vocab_index == 0:
                self.representation[vocab_index] = self._get_pad_embedding()
            elif vocab_index == 1:
                self.representation[vocab_index] = self._get_ukn_embedding()
            else:
                if (self.representation[vocab_index] == zero).all():
                    self.representation[vocab_index] = self._get_other_embedding()
                    self.left_index.append(vocab_index)

        np.savez_compressed(self.trimmed_path, representation=self.representation, left_index=self.left_index,
                            special_index=self.special_index)

    def load_trimmed(self):
        trimmed = np.load(self.trimmed_path)
        self.representation = trimmed['representation']
        self.left_index = trimmed['left_index']
        self.special_index = trimmed['special_index']
        self.len = self.representation.shape[1]

    def get_representation_by_index(self, index):
        return self.representation[index]

    def get_all_representation(self):
        return self.representation

    def get_left_index_token(self):
        return [(index, self.vocab.to_token(index)) for index in self.left_index]

    def _get_pad_embedding(self):
        result = np.random.rand(self.len)
        return result

    def _get_ukn_embedding(self):
        result = np.random.rand(self.len)
        return result

    def _get_other_embedding(self):
        result = np.random.rand(self.len)
        return result


class TextClassifyDataset:

    def __init__(self, file_path, text_length, vocab: Vocab, tokenizer: Tokenizer):
        self.file_path = file_path
        self.text_length = text_length
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.file_iterator = self._get_file_iterator()

    def __iter__(self):
        return self

    def __len__(self):
        return 10 # TODO

    def __next__(self):
        separator = '\t'
        line = next(self.file_iterator)
        sentence, label = line.split(separator)
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) < self.text_length:
            tokens.extend([PAD] * (self.text_length - len(tokens)))
        else:
            tokens = tokens[:self.text_length]
        return [self.vocab.to_index(token) for token in tokens], label

    def _get_file_iterator(self):
        with open(self.file_path, 'r', encoding='UTF-8') as file:
            while True:
                line = file.readline().strip()
                if line == '':
                    break
                else:
                    yield line


class Dataloader:
    def __init__(self, dataset, batch_size):
        self.dataset_iterator = iter(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return 10  # TODO

    def __next__(self):
        i = 0
        X_list = []
        Y_list = []
        while True:
            try:
                X, Y = next(self.dataset_iterator)
            except StopIteration:
                break
            else:
                X_list.append(X)
                Y_list.append(Y)
                i = i + 1
                if i == self.batch_size:
                    break

        if len(X_list) == 0:
            raise StopIteration
        else:
            return X_list, Y_list


# endregion

# region Model
class ModelFactory:
    @staticmethod
    def get_model(config):
        logging.info('--Begin  model.')
        logging.debug(f'\r\n\
                   config.model_name:{config.model_name}\
                ')
        if config.model_name == 'TextCNN':
            import my_ai.model.text_classify
            logging.debug(f'\r\n\
                       config.is_pretrained:{config.is_pretrained}\
                    ')
            model = my_ai.model.text_classify.TextCNN(config)
        else:
            raise Exception("unrecognized model_name!")
        logging.info('--Finished  model.')
        return model


# endregion

# region Trainer
class TrainerFactory:
    @staticmethod
    def get_trainer(config, model):
        if config.model_name == 'TextCNN':
            trainer = TextClassifyTrainer(config, model)
        else:
            raise Exception("unrecognized model_name!")
        return trainer


class TextClassifyTrainer:
    def __init__(self, config: TextClassifyConfig, model):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.animator = my_ai.utility.AnimatorFactory.get_animator()

    def start(self):
        # step1
        self.model.to(self.config.device)
        metric = my_ai.utility.Accumulator(3)  # Sum of training loss, sum of training accuracy, no. of examples
        loss = torch.nn.CrossEntropyLoss()
        timer, num_batches = my_ai.utility.Timer(), len(self.config.train_dataloader)

        self.animator.line_start(num_batches)
        # step2
        for epoch in range(self.config.num_epochs):
            self.model.train()
            for i, (X, y) in enumerate(self.config.train_dataloader):
                # step2.1
                timer.start()
                self.optimizer.zero_grad()
                X, y = torch.tensor(X), torch.tensor(y)
                X, y = X.to(self.config.device), y.to(self.config.device)
                y_hat = self.model(X)
                l = loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                timer.stop()
                # step2.2
                with torch.no_grad():
                    metric.add(l * X.shape[0], my_ai.utility.accuracy(y_hat, y), X.shape[0])
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                self.animator.train_line_append(epoch, i, {"train_l": train_l, "train_acc": train_acc})
            self.evaluate()

    def evaluate(self, epoch):
        metric_eval = my_ai.utility.Accumulator(2)  # No. of correct predictions, no. of predictions
        self.model.eval()
        with torch.no_grad():
            for X, y in self.config.train_dataloader:
                X, y = torch.tensor(X), torch.tensor(y)
                X, y = X.to(self.config.device), y.to(self.config.device)
                metric_eval.add(my_ai.utility.accuracy(self.model(X), y), y.numel())
        test_acc = metric_eval[0] / metric_eval[1]
        self.animator.test_line_append(epoch, {"test_acc": test_acc})

# endregion
