import os

import pandas
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Normalizer:

    def __init__(self, input_size: tuple):
        self.transform_resize = torchvision.transforms.Resize(input_size)

    def normalize(self, x):
        y = self.transform_resize(x)
        return y


class Augmentor:

    def __init__(self): pass

    def rand(self, x):
        # TODO
        return x


class PicClassifyDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, normalizer: Normalizer, augmentor: Augmentor):
        self.img_labels = pandas.read_csv(annotations_file)
        self.img_dir = img_dir
        self.normalizer = normalizer
        self.augmentor = augmentor

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image_int = torchvision.io.read_image(img_path)
        image_int = self.normalizer.normalize(image_int)
        image_int = self.augmentor.rand(image_int)
        image = torch.empty_like(image_int, dtype=torch.float)
        image = image_int / 255
        label = self.img_labels.iloc[idx, 1]
        return image, label


def get_pic_classify_dataloader(annotation_path, img_dir, batch_size, normalizer: Normalizer, augmentor: Augmentor):
    dataset = PicClassifyDataset(annotation_path, img_dir, normalizer, augmentor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
