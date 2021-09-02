import torch
from lib.load import LoadDataset
from lib.model import train_1
from torch import nn


def softmax_regression():
    # step1.load dataset
    batch_size = 256
    train_iter, test_iter = LoadDataset.load_fashion_mnist(batch_size)

    # step2.build model
    net = nn.Sequential(nn.Flatten(), nn.Linear(900, 10))
    num_epochs = 5

    # step3.train model
    train_1(net, train_iter, test_iter, num_epochs, 0.1)
