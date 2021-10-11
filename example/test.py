import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_ai

def test():
    temp = np.load('data/text_classify/embedding.npz')

def train_textCNN():
    conf = my_ai.pipeline.ConfigFactory.get_config('TextCNN','data/text_classify')
    # preprocessor = my_ai.pipeline.PreprocessorFactory.get_preprocessor(conf)
    # dataset = preprocessor.build_dataset(conf)
    # model = my_ai.model.ModelFactory.get_model(conf)
    # trainer = my_ai.pipeline.TrainerFactory.get_trainer(conf, dataset, model)
    # trainer.start()

def predict_textCNN():pass

def run():
    test()


if __name__ == '__main__':
    run()
