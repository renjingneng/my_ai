### Summary

A easy and general DL toolkit based on pytorch.
一个基于pytorch的简单和通用的深度学习工具集。

### Usage

#### train example
```python
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
```
#### inference example
```python
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
```

### Code style

https://numpydoc.readthedocs.io/en/latest/format.html

### Model roadmap

#### text classify(multiple class)

- [X] textCNN
- [ ] TextRNN
- [ ] BiLSTM+Attention
- [ ] TextRCNN
- [ ] Transformer
- [ ] bert

#### text entity recognition

- [ ] bi-lstm-crf

#### picture classify(multiple class)

- [X] improved LeNet