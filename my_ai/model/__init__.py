from . import text_seq
from . import text_classify
from . import pic_classify


class ModelFactory:
    @staticmethod
    def get_model(config):
        model = None
        if config['model_name'] == 'TextCNN':
            model = text_classify.TextCNN(config)
        return model
