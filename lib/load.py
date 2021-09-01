import torchvision
from torchvision import transforms
from torch.utils import data


class LoadDataset:

    @staticmethod
    def __transforms_before_load():
        trans = [transforms.Pad(padding=1), transforms.ToTensor()]
        trans = transforms.Compose(trans)
        return trans

    @staticmethod
    def load_fashion_mnist(batch_size: int):
        trans = LoadDataset.__transforms_before_load()
        """
        Download the Fashion-MNIST dataset and then load it into memory.
        Original dataset uri:https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
        One bug in torchvision 0.10.0 has been fixed in the latest commit: https://github.com/pytorch/vision/pull/4184
        :param batch_size:
        :return:
        """
        mnist_train = torchvision.datasets.FashionMNIST(root="resource/dataset/img",
                                                        train=True,
                                                        transform=trans,
                                                        download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root="resource/dataset/img",
                                                       train=False,
                                                       transform=trans,
                                                       download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=4),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=4))
