import sys
import os
import random
import logging
import configparser

import numpy as np
import torch
import torch.utils.data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_ai.pipeline import ConfigFactory
from my_ai.pipeline import PreprocessorFactory
from my_ai.pipeline import ModelFactory
from my_ai.pipeline import TrainerFactory


def makesure_reproducible():
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.use_deterministic_algorithms(True)


def get_params(conf_path):
    cf = configparser.ConfigParser()
    cf.read(conf_path)
    params_items = cf.items("params")
    if len(params_items) == 0:
        params = None
    else:
        params = {}
        for key, val in params_items:
            params[key] = val
    other_params_items = cf.items("other_params")
    if len(other_params_items) == 0:
        other_params = None
    else:
        other_params = {}
        for key, val in other_params_items:
            other_params[key] = val
    return params, other_params


def train_textCNN():
    # step1.input
    params, other_params = get_params('data/text_classify/config.ini')
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf = ConfigFactory.get_config(params, other_params)
    preprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model = ModelFactory.get_model(conf)
    # step4.train
    trainer = TrainerFactory.get_trainer(conf, model)
    trainer.start()


def predict_textCNN(): pass


def run():
    train_textCNN()


if __name__ == '__main__':
    run()
