import math
import logging
from typing import Union

import jieba
import numpy
import torch
import pickle

UKN, PAD = '<ukn>', '<pad>'
jieba.setLogLevel(log_level=logging.INFO)


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
        seg_list = jieba.cut(text)
        result = list(seg_list)
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
        return self.token_to_idx.get(token, 1)  # if token  not found ,consider it as unknown

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
        self.file_iterator = None

    def __iter__(self):
        self.reset()
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
        self.dataset_iterator.reset()
        return self

    def __len__(self):
        length = len(self.dataset_iterator)
        return math.ceil(length / self.batch_size)

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
            return torch.tensor(X_list), torch.tensor(Y_list)


def get_text_classify_dataloader(file_path, text_length, batch_size, vocab, tokenizer):
    dataset = TextClassifyDataset(file_path, text_length, vocab,
                                  tokenizer)
    dataloader = Dataloader(dataset, batch_size)
    return dataloader
