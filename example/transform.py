import PIL
import torchvision

import lib


class Transforms:
    def __init__(self):
        self.__orig_img = PIL.Image.open('resource/misc/dog_and_cat.jpg')

    def show_pad(self):
        titles = [['Original image', '3', '10', '30'],
                  ['40', '50', '60', '70']]
        imgs = [[torchvision.transforms.Pad(padding=n)(self.__orig_img) for n in (3, 10, 30)],
                [torchvision.transforms.Pad(padding=n)(self.__orig_img) for n in (40, 50, 60, 70)]
                ]
        imgs[0].insert(0, self.__orig_img)
        lib.show_plt.show_image_grid(imgs, titles)

    def show_resize(self):
        titles = [['Original image', '3', '10', '30'],
                  ['40', '50', '60', '70']]
        imgs = [[torchvision.transforms.Resize(size=n)(self.__orig_img) for n in (3, 10, 30)],
                [torchvision.transforms.Resize(size=n)(self.__orig_img) for n in (40, 50, 60, 70)]
                ]
        imgs[0].insert(0, self.__orig_img)
        lib.show_plt.show_image_grid(imgs, titles)
