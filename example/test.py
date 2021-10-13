import sys
import os
import random

import numpy as np
import torch
import torch.utils.data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_ai





def test():
    temp = [1,1,1]
    temp.extend([2,2,2])
    print(temp)


def train_textCNN():
    # np.random.seed(1)
    # random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.use_deterministic_algorithms(True)

    model_name = 'TextCNN'
    files_path = 'data/text_classify'
    conf = my_ai.pipeline.ConfigFactory.get_config(model_name, files_path)
    preprocessor = my_ai.pipeline.PreprocessorFactory.get_preprocessor(conf)
    preprocessor.prepare().get_dataset()
    model = my_ai.model.ModelFactory.get_model(conf)
    trainer = my_ai.pipeline.TrainerFactory.get_trainer(conf, model)
    trainer.start()


def predict_textCNN(): pass


def run():
    train_textCNN()


if __name__ == '__main__':
    run()
