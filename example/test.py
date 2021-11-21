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
import my_ai.pipeline


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
    # ste4.inference
    text = ['词汇阅读是关键 08年考研暑期英语复习全指南', '自考经验谈：自考生毕业论文选题技巧', '本科未录取还有这些路可以走']
    model_manager.load_model()
    result = model_manager.infer(text)
    print(result)


def train_leNet():
    # step1.input
    params, other_params = utility.get_params('data/pic_classify/config.ini')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf: my_ai.pipeline.PicClassifyConfig = ConfigFactory.get_config(params, other_params)
    # conf.show()
    preprocessor: my_ai.pipeline.PicClassifyPreprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model_manager = ModelManager(conf)
    model = model_manager.get_model()
    # utility.summary_of_network(model,[4,1,30,30])
    # step4.train
    trainer = TrainerFactory.get_trainer(conf, model)
    trainer.start()

def predict_leNet():
    # step1.input
    params, other_params = utility.get_params('data/pic_classify/config.ini')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s-%(asctime)s-%(message)s')
    # step2.conf
    conf: my_ai.pipeline.PicClassifyConfig = ConfigFactory.get_config(params, other_params)
    # conf.show()
    preprocessor: my_ai.pipeline.PicClassifyPreprocessor = PreprocessorFactory.get_preprocessor(conf)
    preprocessor.preprocess()
    # step3.model
    model_manager = ModelManager(conf)
    # ste4.inference
    model_manager.load_model()
    img_list = [conf.example_path]
    result = model_manager.infer(img_list)
    print(result)

def run():
    # train_textCNN()
    # predict_textCNN()
    # train_leNet()
    # import torchvision.io
    # image_int = torchvision.io.read_image('data/pic_classify/example.png')
    # image_int2 = torchvision.io.read_image('data/pic_classify/example.png')
    # print(image_int.shape)
    # print(image_int2.shape)
    # img_all = torch.cat([image_int,image_int2],0)
    # print(torch.unsqueeze(img_all,1).shape)
    temp = ['dd','ff']
    print('gg' in temp)
    print('ff' in temp)




if __name__ == '__main__':
    run()
