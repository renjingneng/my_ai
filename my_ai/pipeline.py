import os
import math

import torch
import torch.utils.data
import numpy as np
import pickle as pkl

UKN, PAD = '<ukn>', '<pad>'  # 未知字，padding符号


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
                'total_filters': 256
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
            'total_epochs': 20,
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
        self.original_embed_path = params['files_path'] + '/sgns.sogou.char'
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
        self.total_epochs = params['total_epochs']
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
        # other_params
        self.other_params = other_params


# class TextSeqConfig: pass


# class PicClassifyConfig: pass


# endregion

# region Preprocessor
class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(config):
        if config['model_name'] == 'TextCNN':
            preprocessor = TextClassifyPreprocessor(config)
        else:
            raise Exception("unrecognized model_name!")
        return preprocessor


class TextClassifyPreprocessor:
    def __init__(self, config: TextClassifyConfig):
        self.config = config

    def prepare(self):
        # preprocess config
        self.config.class_list = [x.strip() for x in open(
            self.config.files_path + '/class.txt', encoding='utf-8').readlines()]
        self.config.num_classes = len(self.config.class_list)
        # tokenizer
        self.config.tokenizer = self._get_tokenizer()
        # vocab
        self.config.vocab = self._get_vocab()
        if self.config.is_revocab == 1:
            self.config.vocab.build_vocab()
        else:
            self.config.vocab.load_vocab()
        # embedding
        self.config.embedding = self._get_embedding()
        if self.config.is_retrim_embedding == 1:
            self.config.embedding.build_trimmed()
        else:
            self.config.embedding.load_trimmed()
        return self

    def get_dataset(self):
        train_dataset = TextClassifyDataset(self.config.train_path, self.config.text_length, self.config.vocab,
                                            self.config.tokenizer)
        self.config.train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size)
        dev_dataset = TextClassifyDataset(self.config.dev_path, self.config.text_length, self.config.vocab,
                                          self.config.tokenizer)
        self.config.dev_dataLoader = torch.utils.data.DataLoader(dev_dataset, batch_size=self.config.batch_size)
        text_dataset = TextClassifyDataset(self.config.test_path, self.config.text_length, self.config.vocab,
                                           self.config.tokenizer)
        self.config.text_dataLoader = torch.utils.data.DataLoader(text_dataset, batch_size=self.config.batch_size)
        return self

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
    def __init__(self, train_path, vocab_path, tokenizer, min_freq=2):
        self.train_path = train_path
        self.vocab_path = vocab_path
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        self.idx_to_token, self.token_to_idx = [PAD, UKN], {PAD: 0, UKN: 1}

    def build_vocab(self):
        vocab_dic = {}
        with open(self.train_path, 'r', encoding='UTF-8') as f:
            for line in f:
                sentence = line.strip().split('\t')[0]
                if sentence == '':
                    continue
                for word in self.tokenizer.tokenize(sentence):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1

        i = 2
        for key in vocab_dic:
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

    def to_token(self, index):
        return self.idx_to_token[index]

    def to_index(self, token):
        return self.token_to_idx.get(token, -1)


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
        # embedding_len
        with open(self.original_path, 'r') as f:
            elems = f.readline().strip().split()
            self.len = len(elems[1:])
        # embedding
        self.representation = np.zeros((self.vocab.get_len(), self.len))

        with open(self.original_path, 'r') as f:
            for line in f:
                elems = line.strip().split()
                vocab_index = self.vocab.to_index(elems[0])
                if vocab_index == -1:
                    continue
                if elems[0] == PAD or elems[0] == UKN:
                    continue
                self.representation[vocab_index, 0:] = [float(elem) for elem in elems[1:]]

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

    def _get_pad_embedding(self):
        result = np.zeros(self.len)
        return result

    def _get_ukn_embedding(self):
        result = np.zeros(self.len)
        return result

    def _get_other_embedding(self):
        result = np.random.rand(self.len)
        return result


class TextClassifyDataset(torch.utils.data.IterableDataset):
    """
    loading data async,load one file when single-process,one file per processor when multi-process
    """

    def __init__(self, file_path, text_length, vocab, tokenizer):
        super(TextClassifyDataset).__init__()
        self.file_path = file_path
        self.text_length = text_length
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.file_iterator = None
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.file_iterator = self._get_file_iterator(self.file_path)
        else:
            # TODO
            self.file_iterator = self._get_file_iterator(self.file_path)

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.file_iterator)
        sentence, label = line.split('\t')
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) < self.text_length:
            tokens.extend([PAD] * (self.text_length - len(tokens)))
        else:
            tokens = tokens[:self.text_length]
        return tokens, label

    def _get_file_iterator(self, file_path):
        with open(file_path, 'r') as file:
            while True:
                line = file.readline().strip()
                if line == '':
                    break
                else:
                    yield line


# endregion

# region Trainer
class TrainerFactory:
    @staticmethod
    def get_trainer(config, dataset, model):
        trainer = None
        if config['model_name'] == 'TextCNN':
            trainer = TextClassifyTrainer(dataset, model)
        return trainer


class TextClassifyTrainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def start(self):
        return '222'
# endregion
