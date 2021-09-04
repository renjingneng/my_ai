import collections
import re
import requests
import os


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = collections.Counter(tokens)

        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)

        uniq_tokens = []
        uniq_tokens += reserved_tokens
        for token, freq in self.token_freqs:
            if freq >= min_freq:
                uniq_tokens.append(token)
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


class Treasure:
    def __init__(self, source="timemachine", token_type="word"):
        self.source = source
        self.token_type = token_type
        self.fname = None
        self.__download_from_source()

    def __download_from_source(self):
        if self.source == "timemachine":
            self.fname = f"resource/dataset/text/{self.source}/{self.source}.txt"
            if not os.path.exists(self.fname):
                r = requests.get("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt", stream=True,
                                 verify=True)
                with open(self.fname, 'wb') as f:
                    f.write(r.content)

    def get_vocab(self):
        vocab = Vocab(self.__get_tokens(), 1, ['<unk>', '<pad>', '<bos>', '<eos>'])
        return vocab

    def __line_preprocess(self, line):
        line = re.sub('[^A-Za-z]+', ' ', line).strip().lower()
        return line

    def __token_preprocess(self, token):
        if self.token_type == 'char':
            return token
        if len(token) == 1 and token != "a":
            token = ""
        return token

    def __tokenize(self, line):
        """Split text lines into word or character tokens."""
        if self.token_type == 'word':
            return line.split()
        elif self.token_type == 'char':
            return list(line)
        else:
            raise Exception('ERROR: unknown token type: ' + self.token_type)

    def __get_tokens(self):
        with open(self.fname, 'r') as f:
            raw_lines = f.readlines()
        res = []
        for line in raw_lines:
            line = self.__line_preprocess(line)
            if len(line) == 0:
                continue
            tokens = self.__tokenize(line)
            for token in tokens:
                token = self.__token_preprocess(token)
                if len(token) == 0 and self.token_type == "word":
                    continue
                res.append(token)
        return res
