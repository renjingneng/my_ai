import sys
import os
import logging

import torch
import torch.nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_ai.pipeline import ConfigFactory
from my_ai.pipeline import PreprocessorFactory
from my_ai.pipeline import ModelManager
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
    # step3.model
    model_manager = ModelManager(conf)
    model = model_manager.get_model()
    # step4.train
    trainer = TrainerFactory.get_trainer(conf, model)
    trainer.start()  # 88.69%


def predict_textCNN():
    # step1.input
    params, other_params = utility.get_params('data/text_classify/config.ini')
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf = ConfigFactory.get_config(params, other_params)
    preprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model_manager = ModelManager(conf)
    model = model_manager.get_model()
    # ste4.inference
    text = ['词汇阅读是关键 08年考研暑期英语复习全指南', '自考经验谈：自考生毕业论文选题技巧', '本科未录取还有这些路可以走']
    y_map = ['金融', '现实', '股票', '教育', '科学', '社会', '政治', '体育', '游戏', '娱乐']
    model_manager.load_model()
    result = model_manager.infer(text)
    result = [y_map[item] for item in result]
    print(result)


def run():
    # train_textCNN()
    predict_textCNN()


if __name__ == '__main__':
    run()
