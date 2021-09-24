import sys
import os
import json
import time
import math

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_ai


def test1():
    rnn = nn.LSTM(4, 5, num_layers=1,bidirectional=True)
    input = torch.randn(1, 1, 4)
    h0 = torch.randn(2, 1, 5)
    c0 = torch.randn(2, 1, 5)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output.shape)
    print(hn.shape)
    print(cn.shape)

def test2():
    rnn = nn.LSTM(4, 5, num_layers=1,bidirectional=False)
    input = torch.randn(6, 3, 4)
    h0 = torch.randn(1, 3, 5)
    c0 = torch.randn(1, 3, 5)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output.shape)
    print(hn.shape)
    print(cn.shape)


def run():
    test1()



if __name__ == '__main__':
    run()