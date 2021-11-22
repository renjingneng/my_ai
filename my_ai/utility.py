import time
import random
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


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> int:
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


def makesure_reproducible():
    numpy.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.use_deterministic_algorithms(True)


# endregion

# region File


def save_json_file(obj, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))


def load_json_file(file_path):
    with open(file_path, encoding="utf8") as f:
        return json.load(f)


# endregion
