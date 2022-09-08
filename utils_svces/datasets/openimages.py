import torch
import torch.distributions
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

from .paths import get_openimages_path
from utils_svces.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation
import os

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128

class OpenImages(ImageFolder):
    def __init__(self, root, split, transform=None, target_transform=None, exclude_dataset=None):
        if split == 'train':
            path = os.path.join(root, 'train')
        elif split == 'val':
            path = os.path.join(root, 'val')
        elif split == 'test':
            raise NotImplementedError()
            path = os.path.join(root, 'test')
        else:
            raise ValueError()

        super().__init__(path, transform=transform, target_transform=target_transform)
        exclude_idcs = []

        if exclude_dataset is not None and split == 'train':
            if exclude_dataset == 'imageNet100':
                duplicate_file = 'openImages_imageNet100_duplicates.txt'
            elif exclude_dataset == 'flowers':
                duplicate_file = 'utils/openImages_flowers_idxs.txt'
            elif exclude_dataset == 'pets':
                duplicate_file = 'utils/openImages_pets_idxs.txt'
            elif exclude_dataset == 'cars':
                duplicate_file = 'utils/openImages_cars_idxs.txt'
            elif exclude_dataset == 'food-101':
                duplicate_file = 'utils/openImages_food-101_idxs.txt'
            elif exclude_dataset == 'cifar10':
                print('Warning; CIFAR10 duplicates not checked')
                duplicate_file =  None
            else:
                raise ValueError(f'Exclusion dataset {exclude_dataset} not supported')

            if duplicate_file is not None:
                with open(duplicate_file, 'r') as idxs:
                    for idx in idxs:
                        exclude_idcs.append(int(idx))

        self.exclude_idcs = set(exclude_idcs)
        print(f'OpenImages {split} - Length: {len(self)} - Exclude images: {len(self.exclude_idcs)}')

    def __getitem__(self, index):
        while index in self.exclude_idcs:
            index = np.random.randint(len(self))

        return super().__getitem__(index)

def get_openImages(split='train', batch_size=128, shuffle=None, augm_type='none', num_workers=8, size=224,
                   exclude_dataset=None, config_dict=None):

    augm_config = {}
    transform = get_imageNet_augmentation(type=augm_type, out_size=size, config_dict=augm_config)

    if shuffle is None:
        shuffle = True if split == 'train' else False

    path = get_openimages_path()

    dataset = OpenImages(path, split, transform=transform, exclude_dataset=exclude_dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'OpenImages'
        config_dict['Exclude Dataset'] = exclude_dataset
        config_dict['Length'] = len(dataset)
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader
