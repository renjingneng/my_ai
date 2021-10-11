import torch
import numpy as np


# region Config
class ConfigFactory:
    @staticmethod
    def get_config(model_name, files_path='files', params=None, other_params=None):
        if model_name is None:
            raise Exception("model_name is None!")

        if model_name == "TextCNN":
            default_params = ConfigFactory.get_text_classify_params(model_name, files_path)
            default_other_params = {
                'filter_sizes': (2, 3, 4),
                'total_filters': 256
            }
            params, other_params = ConfigFactory.fill_params(params, other_params, default_params, default_other_params)
            return TextClassifyConfig(params, other_params)
        else:
            raise Exception("unrecognized model_name!")

    @staticmethod
    def fill_params(params, other_params, default_params, default_other_params):
        if params is None:
            params = default_params
        else:
            for key in default_params:
                if params.get(key) is None:
                    params[key] = default_params[key]

        if other_params is None:
            other_params = default_other_params
        else:
            for key in default_other_params:
                if other_params.get(key) is None:
                    other_params[key] = default_other_params[key]
        return params, other_params

    @staticmethod
    def get_text_classify_params(model_name, files_path):
        params = {
            'model_name': model_name,
            'files_path': files_path,
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'is_char_segment': 1,
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
    def __init__(self, params, other_params):
        # file path
        self.train_path = params['files_path'] + '/train.txt'
        self.dev_path = params['files_path'] + '/dev.txt'
        self.test_path = params['files_path'] + '/test.txt'
        self.vocab_path = params['files_path'] + '/vocab.pkl'
        self.save_path = params['files_path'] + '/saved_dict.ckpt'
        self.log_path = params['files_path'] + '/log'

        # basic info
        self.model_name = params['model_name']
        self.device = params['device']
        self.is_char_segment = params['is_char_segment']
        self.is_pretrained = params['is_pretrained']
        self.embedding_length = params['embedding_length']
        self.dropout = params['dropout']
        self.total_epochs = params['total_epochs']
        self.batch_size = params['batch_size']
        self.text_length = params['text_length']
        self.learning_rate = params['learning_rate']
        self.start_expire_after = params['start_expire_after']
        self.expire_batches = params['expire_batches']


        self.class_list = [x.strip() for x in open(
            params['files_path'] + '/class.txt', encoding='utf-8').readlines()]
        self.embedding = torch.tensor(
            np.load(params['files_path'] + '/embedding.npz')["embeddings"].astype('float32')) \
            if params['is_pretrained'] == 1 else None
        self.num_classes = len(self.class_list)


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
            preprocessor = TextClassifyPreprocessor()
        return preprocessor


class TextClassifyPreprocessor:
    def __init__(self):
        self.test = '111'

    def build_dataset(config):
        return '222'

    def preprocess(data):
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
