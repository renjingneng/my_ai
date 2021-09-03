import torch
from lib import load
from lib import model
from lib import summary
from torch import nn
from torch.nn import functional as F


def softmax_regression():
    # step1.load dataset
    batch_size = 256
    train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

    # step2.build model
    net = nn.Sequential(nn.Flatten(), nn.Linear(900, 10))

    # step3.train model
    num_epochs = 5
    model.train_1(net, train_iter, test_iter, num_epochs, lr=0.1)


def original_leNet():
    # step1.load dataset
    batch_size = 128
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
    model.train_1(net, train_iter, test_iter, num_epochs, lr=0.9)


def improved_leNet():
    # step1.load dataset
    batch_size = 128
    load.LoadDataset.transforms_before_load((28, 28))
    train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

    # step2.build model
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
                        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                        nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
    # summary.summary_of_network(net,(1, 1, 28, 28))

    # step3.train model
    num_epochs = 5
    model.train_1(net, train_iter, test_iter, num_epochs, lr=0.1)


def alexNet():
    # step1.load dataset
    batch_size = 128
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
    # summary.summary_of_network(net,(1, 1, 224, 224))

    # step3.train model
    num_epochs = 10
    model.train_1(net, train_iter, test_iter, num_epochs, lr=0.01)


class Vgg11:
    @classmethod
    def vgg_block(cls, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    @classmethod
    def vgg(cls, conv_arch):
        conv_blks = []
        in_channels = 1
        # The convolutional part
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(cls.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # The fully-connected part
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10))

    @classmethod
    def implementation(cls):
        # step1.load dataset
        batch_size = 128
        load.LoadDataset.transforms_before_load((224, 224))
        train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

        # step2.build model
        # Since VGG-11 is more computationally-heavy than AlexNet we construct a network with a
        # smaller number of channels. This is more than sufficient for training on Fashion-MNIST.
        large_conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        ratio = 4
        small_conv_arch = [(pair[0], pair[1] // ratio) for pair in large_conv_arch]
        net = cls.vgg(small_conv_arch)
        # summary.summary_of_network(net,(1, 1, 224, 224))

        # step3.train model
        num_epochs = 10
        model.train_1(net, train_iter, test_iter, num_epochs, lr=0.05)


class NiN:
    @classmethod
    def nin_block(cls, in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU())

    @classmethod
    def implementation(cls):
        # step1.load dataset
        batch_size = 128
        load.LoadDataset.transforms_before_load((224, 224))
        train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

        # step2.build model
        net = nn.Sequential(
            cls.nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            cls.nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            cls.nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
            # There are 10 label classes
            cls.nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # Transform the four-dimensional output into two-dimensional output with a
            # shape of (batch size, 10)
            nn.Flatten())
        # summary.summary_of_network(net, (1, 1, 224, 224))

        # step3.train model
        num_epochs = 10
        model.train_1(net, train_iter, test_iter, num_epochs, lr=0.1)


def googLeNet():
    class Inceptionblock(nn.Module):
        # `c1`--`c4` are the number of output channels for each path
        def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
            super(Inceptionblock, self).__init__(**kwargs)
            # Path 1 is a single 1 x 1 convolutional layer
            self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
            # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
            # convolutional layer
            self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
            self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
            # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
            # convolutional layer
            self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
            self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
            # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
            # convolutional layer
            self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

        def forward(self, x):
            p1 = F.relu(self.p1_1(x))
            p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
            p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
            p4 = F.relu(self.p4_2(self.p4_1(x)))
            # Concatenate the outputs on the channel dimension
            return torch.cat((p1, p2, p3, p4), dim=1)

    # step1.load dataset
    batch_size = 128
    load.LoadDataset.transforms_before_load((96, 96))
    train_iter, test_iter = load.LoadDataset.load_fashion_mnist(batch_size)

    # step2.build model
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2,
                                               padding=1))
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b3 = nn.Sequential(Inceptionblock(192, 64, (96, 128), (16, 32), 32),
                       Inceptionblock(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b4 = nn.Sequential(Inceptionblock(480, 192, (96, 208), (16, 48), 64),
                       Inceptionblock(512, 160, (112, 224), (24, 64), 64),
                       Inceptionblock(512, 128, (128, 256), (24, 64), 64),
                       Inceptionblock(512, 112, (144, 288), (32, 64), 64),
                       Inceptionblock(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(Inceptionblock(832, 256, (160, 320), (32, 128), 128),
                       Inceptionblock(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    # summary.summary_of_network(net, (1, 1, 96, 96))

    # step3.train model
    num_epochs = 10
    model.train_1(net, train_iter, test_iter, num_epochs, lr=0.1)
