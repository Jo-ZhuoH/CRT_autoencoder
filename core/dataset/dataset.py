# -*- coding: utf-8 -*-"
"""
Created on 09/25/2021  1:31 PM


@author: Zhuo
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from core.utils.helper import get_data_paths


class AEDataset(Dataset):
    """ Dataset for autoencoder
    Split the data into training set and validation set.
    """

    def __init__(self, data_paths, transform=None):
        """
        Args:
        :param data_paths: string,
            Directory with all the images.

        :param transform: callable, optional,
            Optional transform to be applied on a sample.

        """
        self.data_paths = data_paths
        self.transform = transform

    def __getitem__(self, index: int):
        image_path = self.data_paths[index]
        image = Image.open(image_path).convert('L')  # Gray scale image
        name = image_path.split("/")[-1].split("_")[0]
        if "pre" in name and len(name) == 8:
            name = name.replace("pre", "pre2")
        sample = {'img': image, 'name': name}

        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    data_path = '/home/zhuo/Desktop/CRT_autoencoder/data/phase/train/GUIDE'
    val_split = 0.0
    idx = 0

    train_paths, _, train_ids, _ = get_data_paths(data_path, val_split)
    tsfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    ds = AEDataset(train_paths, tsfm)

    print(train_paths[idx])
    print(ds.__getitem__(idx)['img'].shape)
