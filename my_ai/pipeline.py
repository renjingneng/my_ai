import math
import logging
import pprint
from typing import Union

import torch
import numpy
import pickle

import my_ai.utility

UKN, PAD = '<ukn>', '<pad>'

"""
different types and related models:
text_classify - TextCNN
pic_classify -
text_entity_extract -
"""


# region Config
class TextClassifyConfig:
    """ok
    """

    @staticmethod
    def get_text_classify_params(model_name: str, files_path: str) -> dict:
        params = {
            'model_name': model_name,
            'files_path': files_path,
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'is_char_segment': True,
            'min_freq': 1,
            'is_revocab': True,
            'is_retrim_embedding': True,
            'is_pretrained': True,
            'embedding_length': 300,
            'dropout': 0.5,
            'num_epochs': 5,
            'batch_size': 128,
            'text_length': 30,
            'learning_rate': 0.003,
            'count_expire_after': 1,  # after how many epochs begin counting expire
            'expire_points': 10,  # early drop after {expire_points} checkpoints without improvement
            'checkpoint_interval': 20,  # check stats after training {checkpoint_interval} batches
        }
        return params

    def __init__(self, params: dict, other_params: dict):
        # files path
        self.files_path = params['files_path']
        self.train_path = params['files_path'] + '/train.txt'
        self.class_path = params['files_path'] + '/class.txt'
        self.dev_path = params['files_path'] + '/dev.txt'
        self.test_path = params['files_path'] + '/test.txt'
        self.vocab_path = params['files_path'] + '/vocab.pkl'
        self.model_save_path = params['files_path'] + '/model_save_path.ckpt'
        self.trimmed_embed_path = params['files_path'] + '/trimmed_embedding.npz'
        self.original_embed_path = params['files_path'] + '/original_embedding'

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
        self.count_expire_after = params['count_expire_after']
        self.expire_points = params['expire_points']
        self.checkpoint_interval = params['checkpoint_interval']

        self.class_list = [x.strip() for x in open(self.class_path, encoding='utf-8').readlines()]
        self.num_classes = len(self.class_list)

        self.tokenizer: Tokenizer = None
        self.vocab: Vocab = None
        self.embedding: Embedding = None
        self.train_dataloader: Dataloader = None
        self.dev_dataloader: Dataloader = None
        self.test_dataloader: Dataloader = None
        # other_params
        self.other_params = other_params

    def show(self):
        pprint.pprint(vars(self))


class ConfigFactory:
    """ok
    """

    @staticmethod
    def get_config(params: dict, other_params: dict = None) -> Union[TextClassifyConfig]:
        if params.get('model_name', None) is None:
            raise Exception("model_name is None!")
        if params.get('files_path', None) is None:
            raise Exception("files_path is None!")

        if params['model_name'] == "TextCNN":
            default_params = TextClassifyConfig.get_text_classify_params(params['model_name'], params['files_path'])
            default_other_params = {
                'filter_sizes': (2, 3, 4),
                'num_filters': 256
            }
            params = ConfigFactory._fill_params(params, default_params)
            other_params = ConfigFactory._fill_params(other_params, default_other_params)
            config = TextClassifyConfig(params, other_params)
        else:
            raise Exception("unrecognized model_name!")
        return config

    @staticmethod
    def _fill_params(params: dict, default_params: dict) -> dict:
        if params is None:
            params = default_params
        else:
            for key in default_params:
                if params.get(key) is None:
                    params[key] = default_params[key]
        return params


# endregion

# region Preprocessor

class TextClassifyPreprocessor:
    """ok
    """

    def __init__(self, config: TextClassifyConfig):
        self.config = config

    def preprocess(self):
        # tokenizer
        logging.info('--Begin  tokenizer.')
        logging.debug(f'\r\n\
           config.is_char_segment:{self.config.is_char_segment}\
        ')
        self.config.tokenizer = Tokenizer(self.config.is_char_segment)
        logging.info('--Finished  tokenizer.')
        # vocab
        logging.info('--Begin  vocab.')
        logging.debug(f'\r\n\
           config.train_path:{self.config.train_path}\r\n\
           config.vocab_path:{self.config.vocab_path}\r\n\
           config.min_freq:{self.config.min_freq}\
        ')
        self.config.vocab = Vocab(self.config.train_path, self.config.vocab_path, self.config.tokenizer,
                                  self.config.min_freq)
        if self.config.is_revocab:
            logging.info('Start building vocab.')
            self.config.vocab.build_vocab_of_sentences()
        else:
            logging.info('Start reloading  vocab.')
            self.config.vocab.load_vocab()
        logging.info('--Finished vocab.')
        # embedding
        if self.config.is_pretrained:
            logging.info('--Begin pretrained  embedding.')
            logging.debug(f'\r\n\
               config.trimmed_embed_path:{self.config.trimmed_embed_path}\r\n\
               config.original_embed_path:{self.config.original_embed_path}\
            ')
            self.config.embedding = Embedding(self.config.trimmed_embed_path, self.config.original_embed_path,
                                              self.config.vocab)
            if self.config.is_retrim_embedding:
                logging.info('Start building pretrained  embedding.')
                self.config.embedding.build_trimmed()
            else:
                logging.info('Start reloading pretrained  embedding.')
                self.config.embedding.load_trimmed()
            logging.debug(f'\r\n\
               residual index with its token:{self.config.embedding.get_residual_index_token()}\
            ')
            logging.info('--Finished pretrained  embedding.')
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


class PreprocessorFactory:
    """ok
    """

    @staticmethod
    def get_preprocessor(config: Union[TextClassifyConfig]) -> Union[TextClassifyPreprocessor]:
        if config.model_name == 'TextCNN':
            preprocessor = TextClassifyPreprocessor(config)
        else:
            raise Exception("unrecognized model_name!")
        return preprocessor


class Tokenizer:
    """ok
    """

    def __init__(self, is_char_segment: bool):
        self.is_char_segment = is_char_segment

    def tokenize(self, text: str) -> list:
        if self.is_char_segment:
            return self.tokenize_by_char(text)
        else:
            return self.tokenize_by_word(text)

    def tokenize_by_char(self, text: str) -> list:
        result = [char for char in text]
        return result

    def tokenize_by_word(self, text: str) -> list:
        result = text.split(' ')
        return result


class Vocab:
    """ok
    """

    def __init__(self, train_path: str, vocab_path: str, tokenizer: Tokenizer, min_freq: int = 2):
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

        pickle.dump(self.idx_to_token, open(self.vocab_path, 'wb'))

    def load_vocab(self):
        self.idx_to_token = pickle.load(open(self.vocab_path, 'rb'))
        self.token_to_idx = {}
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def get_len(self) -> int:
        return len(self.idx_to_token)

    def to_token(self, index: int) -> str:
        if index < 0 or (index + 1) > self.get_len():
            return None
        return self.idx_to_token[index]

    def to_index(self, token: str) -> int:
        return self.token_to_idx.get(token, 1) # if token  not found ,consider it as unknown

    def __len__(self):
        return len(self.idx_to_token)


class Embedding:
    """ok
    """

    def __init__(self, trimmed_path: str, original_path: str, vocab: Vocab):
        self.original_path = original_path
        self.trimmed_path = trimmed_path
        self.vocab = vocab
        self.representation = None  # array  of trimmed Embedding (num of  vocab ,len of Embedding for one token)
        self.len = None  # len of Embedding for one token
        self.residual_index = []  # token index which is not found in original pretrained Embedding
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
        self.representation = numpy.zeros(shape=(self.vocab.get_len(), self.len))

        with open(self.original_path, 'r', encoding='UTF-8') as f:
            if original_has_header:
                f.readline()
            for line in f:
                elems = line.strip().split()
                # if original embedding has PAD or UKN ,just abandon it bc it may has diff meaning
                if elems[0] == PAD or elems[0] == UKN:
                    continue
                vocab_index = self.vocab.to_index(elems[0])
                if vocab_index is None:
                    continue
                self.representation[vocab_index, 0:] = elems[1:]

        zero = numpy.zeros(self.len)
        for vocab_index in range(self.vocab.get_len()):
            if vocab_index == 0:
                self.representation[vocab_index] = self._get_pad_embedding()
            elif vocab_index == 1:
                self.representation[vocab_index] = self._get_ukn_embedding()
            else:
                if (self.representation[vocab_index] == zero).all():
                    self.representation[vocab_index] = self._get_residual_embedding()
                    self.residual_index.append(vocab_index)

        numpy.savez_compressed(self.trimmed_path, representation=self.representation,
                               residual_index=self.residual_index)

    def load_trimmed(self):
        trimmed = numpy.load(self.trimmed_path)
        self.representation = trimmed['representation']
        self.residual_index = trimmed['residual_index']
        self.len = self.representation.shape[1]

    def get_representation_by_index(self, index: int):
        return self.representation[index]

    def get_all_representation(self) -> numpy.ndarray:
        return self.representation

    def get_residual_index(self) -> list:
        return self.residual_index

    def get_residual_index_token(self):
        return [(index, self.vocab.to_token(index)) for index in self.residual_index]

    def _get_pad_embedding(self):
        result = numpy.random.rand(self.len)
        return result

    def _get_ukn_embedding(self):
        result = numpy.random.rand(self.len)
        return result

    def _get_residual_embedding(self):
        result = numpy.random.rand(self.len)
        return result


class TextClassifyDataset:
    """ok
    """

    def __init__(self, file_path: str, text_length: str, vocab: Vocab, tokenizer: Tokenizer):
        self.file_path = file_path
        self.text_length = text_length
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.file_iterator = self._get_file_iterator()

    def __iter__(self):
        return self

    def __len__(self):
        length = 0
        with open(self.file_path, 'r', encoding='UTF-8') as file:
            while True:
                line = file.readline().strip()
                if line == '':
                    break
                else:
                    length = length + 1
        return length

    def __next__(self):
        separator = '\t'
        line = next(self.file_iterator)
        sentence, label = line.split(separator)
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) < self.text_length:
            tokens.extend([PAD] * (self.text_length - len(tokens)))
        else:
            tokens = tokens[:self.text_length]
        return [self.vocab.to_index(token) for token in tokens], int(label)

    def reset(self):
        self.file_iterator = self._get_file_iterator()
        return self

    def _get_file_iterator(self):
        with open(self.file_path, 'r', encoding='UTF-8') as file:
            while True:
                line = file.readline().strip()
                if line == '':
                    break
                else:
                    yield line


class Dataloader:
    """ok
    """

    def __init__(self, dataset_iterator: Union[TextClassifyDataset], batch_size: int):
        self.dataset_iterator = dataset_iterator
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __len__(self):
        length = len(self.dataset_iterator)
        return math.ceil(length / self.batch_size)

    def reset(self):
        self.dataset_iterator.reset()
        return

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
    """ok
    """

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
    """ok
    """

    @staticmethod
    def get_trainer(config, model):
        logging.info('--Begin  trainer.')
        logging.debug(f'\r\n\
                   config.num_epochs:{config.num_epochs}\r\n\
                   config.count_expire_after:{config.count_expire_after}\r\n\
                   config.checkpoint_interval:{config.checkpoint_interval}\
                ')
        if config.model_name == 'TextCNN':
            trainer = TextClassifyTrainer(config, model)
        else:
            raise Exception("unrecognized model_name!")
        logging.info('--Finished  trainer.')
        return trainer


class TextClassifyTrainer:
    def __init__(self, config: TextClassifyConfig, model):
        """ok
        """
        # essential components
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.animator = my_ai.utility.get_animator()
        self.loss_func = torch.nn.CrossEntropyLoss()
        # important stats
        self.num_batches = len(self.config.train_dataloader)
        self.dev_best_loss = float('inf')
        self.last_improve_point = -1
        self.now_point = -1
        self.is_expire = False
        self.epoch = -1

    def start(self):
        self.model.to(self.config.device)
        self.animator.prepare(self.num_batches)

        for _ in range(self.config.num_epochs):
            self.epoch = self.epoch + 1
            # train
            is_continue = self.train()
            if not is_continue:
                break
            # evaluate
            self.evaluate()

    def train(self):
        metric = my_ai.utility.Accumulator(3)  # Sum of training loss, sum of training accuracy, no. of examples
        self.config.train_dataloader.reset()
        self.model.train()

        for i, (X, y) in enumerate(self.config.train_dataloader):
            # forward backward
            self.optimizer.zero_grad()
            X, y = torch.tensor(X), torch.tensor(y)
            X, y = X.to(self.config.device), y.to(self.config.device)
            y_hat = self.model(X)
            l = self.loss_func(y_hat, y)
            l.backward()
            self.optimizer.step()
            # checkpoint
            if i != 0 and i % self.config.checkpoint_interval == 0:
                self.now_point = self.now_point + 1
                self.checkpoint()
            # recording stats
            with torch.no_grad():
                metric.add(l * X.shape[0], my_ai.utility.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            self.animator.train_line_append(self.epoch, i, {"train_l": train_l, "train_acc": train_acc})
            # condition for early stop
            if self.epoch >= self.config.count_expire_after and self.is_expire:
                self.log_action('Early stop')
                return False

        return True

    def checkpoint(self):
        dev_loss = self.get_dev_loss()
        if dev_loss < self.dev_best_loss:
            self.save()
            self.last_improve_point = self.now_point
            self.dev_best_loss = dev_loss
        else:
            if (self.now_point - self.last_improve_point) >= self.config.expire_points:
                self.is_expire = True
        return self

    def save(self):
        self.log_action('Save model')
        torch.save(self.model.state_dict(), self.config.model_save_path)
        return self

    def log_action(self, action: str):
        action_statement = 'epoch:{} , now_point:{} ,action:{} '.format(self.epoch, self.now_point, action)
        logging.info(action_statement)
        return self

    def get_dev_loss(self):
        metric = my_ai.utility.Accumulator(2)  # sum of   loss,number of examples
        self.model.eval()
        with torch.no_grad():
            for X, y in self.config.dev_dataloader:
                X, y = torch.tensor(X), torch.tensor(y)
                X, y = X.to(self.config.device), y.to(self.config.device)
                y_hat = self.model(X)
                l = self.loss_func(y_hat, y)  # average   loss of this batch
                metric.add(l * X.shape[0], X.shape[0])
        dev_loss = metric[0] / metric[1]
        return dev_loss

    def evaluate(self):
        metric = my_ai.utility.Accumulator(2)  # No. of correct predictions, no. of predictions
        self.model.eval()
        with torch.no_grad():
            for X, y in self.config.test_dataloader:
                X, y = torch.tensor(X), torch.tensor(y)
                X, y = X.to(self.config.device), y.to(self.config.device)
                y_hat = self.model(X)
                metric.add(my_ai.utility.accuracy(y_hat, y), y.numel())
        test_acc = metric[0] / metric[1]
        self.animator.test_line_append(self.epoch, {"test_acc": test_acc})
        return self

# endregion
