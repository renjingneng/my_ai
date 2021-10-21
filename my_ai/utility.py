import time
import json
from abc import ABC, abstractmethod

import torch
import numpy
import torchinfo
import matplotlib.pyplot
import visdom


# region Animator
class Animator(ABC):
    @abstractmethod
    def prepare(self, num_batches): pass

    @abstractmethod
    def train_line_append(self, epoch, batch, data): pass

    @abstractmethod
    def test_line_append(self, epoch, data, batch=None): pass


class AnimatorVisdom(Animator):
    def __init__(self):
        self.vis = visdom.Visdom()
        self.num_batches = None

    def prepare(self, num_batches):
        self.num_batches = num_batches
        self.vis.close(win="train_line")
        self.vis.close(win="test_line")
        self.vis.line(X=[0], Y=[0], win="train_line", name="train_l", update='append')
        self.vis.line(X=[0], Y=[0], win="train_line", name="train_acc", update='append')
        self.vis.line(X=[0], Y=[0], win="test_line", name="test_acc", update='append')

    def train_line_append(self, epoch, batch, data):
        """
        :param epoch: start from 0,show start from 1
        :param batch: start from 0
        :param data: {train_l,train_acc}
        :return:
        """
        train_l = data["train_l"]
        train_acc = data["train_acc"]
        self.vis.line(X=[epoch + (batch + 1) / self.num_batches], Y=[train_l], win="train_line", name="train_l",
                      update='append')
        self.vis.line(X=[epoch + (batch + 1) / self.num_batches], Y=[train_acc], win="train_line", name="train_acc",
                      update='append')

    def test_line_append(self, epoch, data, batch=None):
        """
        :param epoch: start from 0,show start from 1
        :param data: {test_acc}
        :param batch: start from 0
        :return:
        """
        test_acc = data["test_acc"]
        if batch:
            self.vis.line(X=[epoch + (batch + 1) / self.num_batches], Y=[test_acc], win="test_line", name="test_acc",
                          update='append')
        else:
            self.vis.line(X=[epoch + 1], Y=[test_acc], win="test_line", name="test_acc", update='append')


def get_animator(animator_type="visdom"):
    if animator_type == "visdom":
        return AnimatorVisdom()


# endregion

# region Calculation
def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    https://mc-stan.org/docs/2_27/stan-users-guide/log-sum-of-exponentials.html
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


# endregion

# region Metrics
class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self.tik = 0

    def start(self):
        self.tik = time.time()

    def stop(self):
        now = time.time()
        self.times.append(time.time() - self.tik)
        self.tik = now
        return self.times[-1]

    def end(self):
        self.times.append(time.time() - self.tik)
        self.tik = 0
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return numpy.array(self.times).cumsum().tolist()


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) == 2:
        y_hat = y_hat.argmax(axis=1)
    elif len(y_hat.shape) > 2:
        raise Exception
    y = y.type(y_hat.dtype)
    cmp = y_hat == y
    return cmp.sum().item()


def summary_of_network(net, input_size):
    """
    show summary of network
    :param net:
    :param input_size:
    :return:
    """
    torchinfo.summary(net, input_size, verbose=2, col_names=["output_size", "num_params", "mult_adds"])


# endregion

# region NNBlock
def cnn_block(num_convs, in_channels, out_channels):
    """
    :param num_convs:
    :param in_channels:
    :param out_channels:
    :return:
    """
    layers = []
    for _ in range(num_convs):
        layers.append(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        in_channels = out_channels
    layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    return torch.nn.Sequential(*layers)


# endregion

# region Train
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def train_1(net, train_iter, test_iter, num_epochs, lr):
    """
    Layer: Linear && Conv2d
    Optim: SGD
    Loss: CrossEntropyLoss
    Purpose: img classification
    """
    # step1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.apply(init_weights)
    train_l = 0
    train_acc = 0
    test_acc = 0
    metric = Accumulator(3)  # Sum of training loss, sum of training accuracy, no. of examples
    metric_eval = Accumulator(2)  # No. of correct predictions, no. of predictions
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    timer, num_batches = Timer(), len(train_iter)
    # vis = visdom.Visdom()
    # vis.close(win="lineset")
    # vis.line(X=[0], Y=[train_l], win="lineset", name="train_l", update='append')
    # vis.line(X=[0], Y=[train_acc], win="lineset", name="train_acc", update='append')
    # vis.line(X=[0], Y=[test_acc], win="lineset", name="test_acc", update='append')
    animator = get_animator()
    animator.prepare(num_batches)
    # step2
    for epoch in range(num_epochs):
        net.train()
        for i, (X, y) in enumerate(train_iter):
            # step2.1
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            timer.stop()
            # step2.2
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            animator.train_line_append(epoch, i, {"train_l": train_l, "train_acc": train_acc})
        # step2.3
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(device)
                y = y.to(device)
                metric_eval.add(accuracy(net(X), y), y.numel())
        test_acc = metric_eval[0] / metric_eval[1]
        animator.test_line_append(epoch, {"test_acc": test_acc})


# endregion

# region Other
def show_image_grid(imgs, titles):
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = matplotlib.pyplot.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(numpy.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[row_idx, col_idx].set(title=titles[row_idx][col_idx])
            axs[row_idx, col_idx].title.set_size(8)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()


def save_json_file(obj, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))


def load_json_file(file_path):
    with open(file_path, encoding="utf8") as f:
        return json.load(f)

# endregion
