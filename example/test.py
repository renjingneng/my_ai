import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_ai.pipeline import ConfigFactory
from my_ai.pipeline import PreprocessorFactory
from my_ai.pipeline import ModelFactory
from my_ai.pipeline import TrainerFactory
import my_ai.utility as utility


def train_textCNN():
    # step1.input
    params, other_params = utility.get_params('data/text_classify/config.ini')
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf = ConfigFactory.get_config(params, other_params)
    preprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    return
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
