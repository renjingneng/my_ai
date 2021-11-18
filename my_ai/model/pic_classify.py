import torch
import torch.nn as nn

import my_ai.pipeline
import my_ai.model


class LeNet(torch.nn.Module):
    """
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
                                     nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                                     nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))

    def forward(self, x):
        result = self.network(x)
        return result
