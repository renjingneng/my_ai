import sys
import os
import logging

import torch
import torch.nn
from torchvision.io import read_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_ai.pipeline import ConfigFactory
from my_ai.pipeline import PreprocessorFactory
from my_ai.pipeline import ModelManager
from my_ai.pipeline import TrainerFactory
import my_ai.utility as utility
import my_ai.pipeline as pipeline


def train_textCNN():
    # step1.input
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf: pipeline.TextClassifyConfig = ConfigFactory.get_config('data/text_classify/config.ini')
    preprocessor: pipeline.TextClassifyPreprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model_manager = ModelManager(conf)
    model = model_manager.get_model()
    # step4.train
    trainer: pipeline.TextClassifyTrainer = TrainerFactory.get_trainer(conf, model)
    trainer.start()  # 88.69%


def predict_textCNN():
    # step1.input
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf: pipeline.TextClassifyConfig = ConfigFactory.get_config('data/text_classify/config.ini')
    preprocessor: pipeline.TextClassifyPreprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model_manager = ModelManager(conf)
    # ste4.inference
    text = ['词汇阅读是关键 08年考研暑期英语复习全指南', '自考经验谈：自考生毕业论文选题技巧', '本科未录取还有这些路可以走']
    model_manager.load_model()
    result = model_manager.infer(text)
    print(result)


def train_leNet():
    # step1.input
    logging.basicConfig(level=logging.INFO, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf: pipeline.PicClassifyConfig = ConfigFactory.get_config('data/pic_classify/config.ini')
    # conf.show()
    preprocessor: pipeline.PicClassifyPreprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model_manager = ModelManager(conf)
    model = model_manager.get_model()
    # utility.summary_of_network(model,[4,1,28,28])
    # step4.train
    trainer: pipeline.PicClassifyTrainer = TrainerFactory.get_trainer(conf, model)
    trainer.start()


def predict_leNet():
    # step1.input
    logging.basicConfig(level=logging.INFO, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf: pipeline.PicClassifyConfig = ConfigFactory.get_config('data/pic_classify/config.ini')
    # conf.show()
    # step3.model
    model_manager = ModelManager(conf)
    # ste4.inference
    model_manager.load_model()
    # img_list = ['data/pic_classify/test2.png','data/pic_classify/test4.png','data/pic_classify/test6.png','data/pic_classify/test7.png']
    img_list = [
                'data/pic_classify/example/example0.jpg',
                'data/pic_classify/example/example1.jpg',
                'data/pic_classify/example/example2.jpg',
                'data/pic_classify/example/example3.jpg',
                'data/pic_classify/example/example4.jpg',
                'data/pic_classify/example/example6.jpg',
                'data/pic_classify/example/example7.jpg',
                'data/pic_classify/example/example8.jpg',
                'data/pic_classify/example/example9.jpg',
                ]
    result = model_manager.infer(img_list)
    print(result)


def run():
    # train_textCNN()
    # predict_textCNN()
    # train_leNet()
     predict_leNet()


if __name__ == '__main__':
    run()
