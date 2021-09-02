import visdom
from abc import ABC, abstractmethod


class Animator(ABC):
    @abstractmethod
    def line_start(self, num_batches): pass

    @abstractmethod
    def train_line_append(self, epoch, batch, data): pass

    @abstractmethod
    def test_line_append(self, epoch, data, batch=None): pass


class AnimatorVisdom(Animator):
    def __init__(self):
        self.vis = None
        self.num_batches = None

    def line_start(self, num_batches):
        self.num_batches = num_batches
        self.vis = visdom.Visdom()
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


class AnimatorFactory:
    @staticmethod
    def get_animator(animator_type="visdom"):
        if animator_type == "visdom":
            return AnimatorVisdom()
