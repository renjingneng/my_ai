import lib.model
import torch


def run():
    # example1 = example.transform.Transforms()
    # example1.show_resize()
    # net = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    #                     nn.AvgPool2d(kernel_size=2, stride=2),
    #                     nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    #                     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    #                     nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    #                     nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
    # lib.show_cmd.show_network(net,(2,3,28,28))
    # train, test = lib.load.LoadDataset.load_fashion_mnist(256)
    temp1 = torch.tensor([1, 1, 1])
    temp2 = torch.tensor([1, 1, 0])
    print(lib.model.accuracy(temp1, temp2))


if __name__ == '__main__':
    run()
