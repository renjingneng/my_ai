import torchvision
import torch
import torch.utils.data


class LoadDataset:
    trans = None

    @classmethod
    def transforms_before_load(cls, resize=None):
        cls.trans = [torchvision.transforms.ToTensor()]
        if resize:
            cls.trans.insert(0, torchvision.transforms.Resize(resize))
        cls.trans = torchvision.transforms.Compose(cls.trans)
        return cls.trans

    @classmethod
    def load_fashion_mnist(cls, batch_size: int):
        if not cls.trans:
            trans = cls.transforms_before_load()
        else:
            trans = cls.trans
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
        return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                            num_workers=4),
                torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                            num_workers=4))
