import torchvision
from torchvision import transforms
from torch.utils import data



def load_data_fashion_mnist(batch_size:int, resize:int=None):
    """
    Download the Fashion-MNIST dataset and then load it into memory.
    :param batch_size:
    :param resize:
    :return:
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="resource/dataset/classification",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="resource/dataset/classification",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4))
