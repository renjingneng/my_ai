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


def alexNet():
    # step1.load dataset
    batch_size = 256
    load.LoadDataset.transforms_before_load((224, 224))
    train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

    # step2.build model
    net = nn.Sequential(
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of the
        # output. Here, the number of output channels is much larger than that in
        # LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the number of
        # output channels
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of output
        # channels is further increased. Pooling layers are not used to reduce the
        # height and width of input after the first two convolutional layers
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of classes is
        # 10, instead of 1000 as in the paper
        nn.Linear(4096, 10))
    # summary.summary_of_network(net,(20, 3, 224, 224))

    # step3.train model
    num_epochs = 10
    model.train_1(net, train_iter, test_iter, num_epochs, 0.01)


class Vgg11:
    @staticmethod
    def vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    @staticmethod
    def vgg(conv_arch):
        conv_blks = []
        in_channels = 1
        # The convolutional part
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(Vgg11.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # The fully-connected part
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10))

    @staticmethod
    def implementation():
        # step1.load dataset
        batch_size = 256
        load.LoadDataset.transforms_before_load((224, 224))
        train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

        # step2.build model
        # Since VGG-11 is more computationally-heavy than AlexNet we construct a network with a
        # smaller number of channels. This is more than sufficient for training on Fashion-MNIST.
        large_conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        ratio = 4
        small_conv_arch = [(pair[0], pair[1] // ratio) for pair in large_conv_arch]
        net = Vgg11.vgg(small_conv_arch)
        # summary.summary_of_network(net,(1, 1, 224, 224))

        # step3.train model
        num_epochs = 10
        model.train_1(net, train_iter, test_iter, num_epochs, 0.05)
