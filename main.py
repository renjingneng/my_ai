import example.cnn_model
import lib.model
import torch
import visdom
import numpy as np
import time
import numpy
from torch import nn
from d2l import torch as d2l
from torchvision.transforms.functional import resize


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

    # temp1 = torch.tensor([1, 1, 1])
    # temp2 = torch.tensor([1, 1, 0])
    # print(lib.model.accuracy(temp1, temp2))

    # vis = visdom.Visdom()
    # vis.close("lineset")
    # X1=[0]
    # Y1=[0]
    # Y2=[4]
    # Y3=[6]
    # vis.line(X=X1,Y=Y1,win="lineset",name="line1",update='append')
    # vis.line(X=X1,Y=Y2,win="lineset",name="line2",update='append')
    # vis.line(X=X1,Y=Y3,win="lineset",name="line3",update='append')
    # for i in range(10):
    #     time.sleep(1)
    #     X1 = [i+1]
    #     Y1 = [i+1]
    #     vis.line(X=X1,Y=Y1,win="lineset",name="line1",update='append')
    #     vis.line(X=X1,Y=Y2,win="lineset",name="line2",update='append')
    #     vis.line(X=X1,Y=Y3,win="lineset",name="line3",update='append')
    # print('debug')
    example.cnn_model.googLeNet()
    # print(resize(torch.ones(1,3,5,5),(4,4)))


if __name__ == '__main__':
    run()
