import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

"""
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#loss-functions
"""

imsize = 512 if torch.cuda.is_available() else 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])
unloader = transforms.ToPILImage()  # reconvert into PIL image


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    plt.ion()
    plt.figure()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.ioff()
    plt.pause(2)  # pause a bit so we can see what this pic is


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def run():
    style_img = image_loader("./picasso.jpg")
    content_img = image_loader("./dancing.jpg")
    imshow(style_img, title='Style Image')
    imshow(content_img, title='Content Image')
    plt.show()


if __name__ == '__main__':
    run()
