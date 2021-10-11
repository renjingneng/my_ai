import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_ai


def run():
    conf = my_ai.pipeline.ConfigFactory.get_config('TextCNN')
    preprocessor = my_ai.pipeline.PreprocessorFactory.get_preprocessor(conf)
    dataset = preprocessor.build_dataset(conf)
    model = my_ai.model.ModelFactory.get_model(conf)
    trainer = my_ai.pipeline.TrainerFactory.get_trainer(conf, dataset, model)
    trainer.start()


if __name__ == '__main__':
    run()
