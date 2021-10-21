import sys
import os
import random
import logging

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


def train_textCNN():
    # step1.input
    model_name = 'TextCNN'
    files_path = 'data/text_classify'
    params = {'is_revocab': 0, 'is_retrim_embedding': 0, 'min_freq': 1}
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf = ConfigFactory.get_config(model_name, files_path, params)
    preprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model = ModelFactory.get_model(conf)
    # step4.train
    trainer = TrainerFactory.get_trainer(conf, model)
    # trainer.start()


def predict_textCNN(): pass


def run():
    train_textCNN()


if __name__ == '__main__':
    run()
