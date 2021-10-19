import sys
import os
import random

import numpy as np
import torch
import torch.utils.data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_ai


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
    # step2.conf
    conf = my_ai.pipeline.ConfigFactory.get_config(model_name, files_path)
    preprocessor = my_ai.pipeline.PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model = my_ai.model.ModelFactory.get_model(conf)
    # step4.train
    trainer = my_ai.pipeline.TrainerFactory.get_trainer(conf, model)
    trainer.start()


def predict_textCNN(): pass


def run():
    train_textCNN()


if __name__ == '__main__':
    run()
