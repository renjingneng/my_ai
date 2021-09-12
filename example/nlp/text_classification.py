import os
import sys
import numpy as np
import tarfile
import wget
import warnings
from zipfile import ZipFile
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant



class Example:
    def __init__(self):
        warnings.filterwarnings("ignore")
        GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
        TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'aclImdb/train')
        TEST_DATA_DIR = os.path.join(BASE_DIR, 'aclImdb/test')