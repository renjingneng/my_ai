import time
import json
from abc import ABC, abstractmethod

import torch
import numpy as np
import torchinfo
import matplotlib.pyplot
import visdom


# region Config
class ConfigFactory:
    @staticmethod
    def get_config(model_name, files_path='files', common_hyperparams=None, other_hyperparams=None):
        if model_name is None:
            raise Exception("model_name is None")

        if model_name == "TextCNN":
            if common_hyperparams is None: common_hyperparams = ConfigFactory.get_text_classify_common_hyperparams()
            if other_hyperparams is None: other_hyperparams = {
                'dd': 'fff'
            }
            return TextClassifyConfig(files_path, common_hyperparams, other_hyperparams)

    @staticmethod
    def get_text_classify_common_hyperparams():
        common_hyperparams = {
            'epochs': 5,
            'text_length': 30,
            'learning_rate': 0.005,
        }
        return common_hyperparams


class TextClassifyConfig:
    def __init__(self, params, other_params):
        # files_path
        self.train_path = files_path + '/data/train.txt'
        self.validate_path = files_path + '/data/validate.txt'
        self.test_path = files_path + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            files_path + '/data/class.txt', encoding='utf-8').readlines()]
        self.vocab_path = files_path + '/data/vocab.pkl'
        self.save_path = files_path + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = files_path + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(files_path + '/data/' + common_hyperparams['embedding'])["embeddings"].astype('float32')) \
            if common_hyperparams['embedding'] != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # model common
        self.dropout = common_hyperparams['dropout']
        self.require_improvement = common_hyperparams['require_improvement']
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0
        self.epochs = common_hyperparams['epochs']
        self.batch_size = common_hyperparams['batch_size']
        self.text_length = common_hyperparams['text_length']
        self.learning_rate = common_hyperparams['learning_rate']
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        # model specific
        self.other_params = other_params


# class TextSeqConfig: pass


# class PicClassifyConfig: pass


# endregion

# region Preprocessor
class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(config):
        if config['model_name'] == 'TextCNN':
            preprocessor = TextClassifyPreprocessor()
        return preprocessor


class TextClassifyPreprocessor:
    def __init__(self):
        self.test = '111'

    def build_dataset(config):
        return '222'


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
