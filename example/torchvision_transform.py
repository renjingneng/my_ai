import sys
import os

import PIL
import torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_ai.utility


class Transforms:
    def __init__(self,pic_path:str):
        self.__orig_img = PIL.Image.open(pic_path)

    def show_pad(self):
        titles = [['Original image', '3', '10', '30'],
                  ['40', '50', '60', '70']]
        imgs = [[torchvision.transforms.Pad(padding=n)(self.__orig_img) for n in (3, 10, 30)],
                [torchvision.transforms.Pad(padding=n)(self.__orig_img) for n in (40, 50, 60, 70)]
                ]
        imgs[0].insert(0, self.__orig_img)
        my_ai.utility.show_image_grid(imgs, titles)

    def show_resize(self):
        titles = [['Original image', '3', '10', '28'],
                  ['40', '50', '60', '70']]
        imgs = [[torchvision.transforms.Resize(size=n)(self.__orig_img) for n in (3, 10, 30)],
                [torchvision.transforms.Resize(size=n)(self.__orig_img) for n in (40, 50, 60, 70)]
                ]
        imgs[0].insert(0, self.__orig_img)
        my_ai.utility.show_image_grid(imgs, titles)

def run():
    example = Transforms('data/pic_classify/example4.png')
    example.show_resize()

if __name__ == '__main__':
    run()