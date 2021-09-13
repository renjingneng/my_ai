import collections
import re
import os
import torch
from torch.utils import data

"""
tips:
 embedding:
    For padding, fill a zero vector embedding .
    For words that don't have a pre-trained embedding, you should fill them with random values when initializing,
    but set them to trainable.
"""


class WordEmbedding:
    def __init__(self, embedding_name='glove.6B.50d'):
        self.data_path = self.get_data_path(embedding_name)
        self.word_to_vec = {"<zero>": None}
        self.embedding_len = None

        with open(self.data_path, 'r') as f:
            elems = f.readline().strip().split()
            self.embedding_len = len(elems[1:])
            self.word_to_vec["<zero>"] = [0.] * self.embedding_len

        with open(self.data_path, 'r') as f:
            for line in f:
                elems = line.strip().split()
                self.word_to_vec[elems[0]] = [float(elem) for elem in elems[1:]]

    def get_data_path(self, embedding_name):
        if embedding_name == "glove.6B.50d":
            return "resource/word_vector/glove_en/glove.6B/glove.6B.50d.txt"

    def __getitem__(self, words):
        if isinstance(words, list):
            vecs = [
                self.word_to_vec.get(word, self.word_to_vec["<zero>"])
                for word in words]
            return vecs
        else:
            return self.word_to_vec.get(words, self.word_to_vec["<zero>"])

    def __len__(self):
        return len(self.word_to_vec)


class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # format:counter -> Counter({'dd': 2, 'ffff': 1})
        counter = collections.Counter(tokens)
        # format:self.token_freqs -> [('dd', 2), ('faff', 1)]
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        uniq_tokens = []
        uniq_tokens += reserved_tokens
        for token, freq in self.token_freqs:
            if freq >= min_freq:
                uniq_tokens.append(token)
        self.idx_to_token, self.token_to_idx = [], dict()
        for index, token in enumerate(uniq_tokens):
            self.idx_to_token.append(token)
            self.token_to_idx[token] = index

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.token_to_idx.get(token, 0) for token in tokens]
        else:
            return self.token_to_idx.get(tokens, 0)

    def to_tokens(self, indices):
        if isinstance(indices, (list, tuple)):
            return [self.idx_to_token[index] for index in indices]
        else:
            return self.idx_to_token[indices]


class DatasetFactory(object):
    @staticmethod
    def get_dataset(name):
        if name == "aclImdb":
            return DatasetAclImdb()
        elif name == "中文语料库-分类":
            return DatasetTestCh()


class DatasetClassification(object):
    def read_data(self, data_dir, is_train):
        data, labels, label_map = [], [], {}
        data_folder = os.path.join(data_dir, 'train' if is_train else 'test')
        for index, label in enumerate(os.listdir(data_folder)):
            label_map[index] = label
            label_folder = os.path.join(data_folder,
                                        label)
            for file in os.listdir(label_folder):
                with open(os.path.join(label_folder, file), 'rb') as f:
                    for line in f:
                        line = line.decode('utf-8')
                        if line.strip() == "":
                            continue
                        line = self.line_preprocess(line)
                        data.append(self.line_tokenize(line))
                        labels.append(index)
        return data, labels, label_map

    def truncate_pad(self, line, num_steps, padding_token):
        if len(line) > num_steps:
            return line[:num_steps]  # Truncate
        return line + [padding_token] * (num_steps - len(line))  # Pad

    def line_preprocess(self):
        raise NotImplementedError

    def line_tokenize(self):
        raise NotImplementedError

    def token_preprocess(self):
        raise NotImplementedError


class DatasetClassificationEn(DatasetClassification):
    def line_preprocess(self, line):
        line = re.sub('[^A-Za-z]+', ' ', line).strip().replace('\n', '').lower()
        return line

    def line_tokenize(self, line):
        tokens = line.split()
        res = []
        for token in tokens:
            token = self.token_preprocess(token)
            if token:
                res.append(token)
        return res

    def token_preprocess(self, token):
        if len(token) == 1 and token != "a":
            token = None
        return token


class DatasetClassificationCh(DatasetClassification):
    pass


class DatasetAclImdb(DatasetClassificationEn):

    def __init__(self, num_steps=10, batch_size=24):
        self.vocab = None
        self.train_iter = None
        self.test_iter = None

        data_path = f"resource/dataset/text/aclImdb"
        train_data = self.read_data(data_path, True)
        test_data = self.read_data(data_path, False)
        train_tokens = []
        for example in train_data[0]:
            train_tokens += example
        self.vocab = Vocab(train_tokens, 5, ["<pad>", "<ukn>"])
        train_features = torch.tensor([
            self.truncate_pad(self.vocab[example], num_steps, self.vocab['<pad>'])
            for example in train_data[0]])
        test_features = torch.tensor([
            self.truncate_pad(self.vocab[example], num_steps, self.vocab['<pad>'])
            for example in test_data[0]])
        self.train_iter = data.DataLoader(data.TensorDataset(train_features, torch.tensor(train_data[1])), batch_size,
                                          shuffle=True)
        self.test_iter = data.DataLoader(data.TensorDataset(test_features, torch.tensor(test_data[1])), batch_size,
                                         shuffle=False)


class DatasetTestCh(DatasetClassificationCh):
    pass
