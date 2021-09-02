import torch
from lib import load
from lib import model
from lib import summary
from torch import nn


def softmax_regression():
    # step1.load dataset
    batch_size = 256
    train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

    # step2.build model
    net = nn.Sequential(nn.Flatten(), nn.Linear(900, 10))

    # step3.train model
    num_epochs = 5
    model.train_1(net, train_iter, test_iter, num_epochs, 0.1)


def original_leNet():
    # step1.load dataset
    batch_size = 256
    load.LoadDataset.transforms_before_load((28, 28))
    train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

    # step2.build model
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                        nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
    # summary.summary_of_network(net,(20, 1, 28, 28))

    # step3.train model
    num_epochs = 5
    model.train_1(net, train_iter, test_iter, num_epochs, 0.9)


def improved_leNet():
    # step1.load dataset
    batch_size = 256
    load.LoadDataset.transforms_before_load((28, 28))
    train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

    # step2.build model
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
                        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                        nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
    # summary.summary_of_network(net,(20, 1, 28, 28))

    # step3.train model
    num_epochs = 5
    model.train_1(net, train_iter, test_iter, num_epochs, 0.1)
