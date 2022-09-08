import torch
import torch.distributions
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

from .paths import get_CINIC10_path
from utils_svces.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128


def get_CINIC10(split='train', batch_size=None, shuffle=False,
                  augm_type='none', cutout_window=16, num_workers=2, size=32, config_dict=None):
    if batch_size == None:
        batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window, out_size=size, config_dict=augm_config)

    path = get_CINIC10_path()
    if split == 'train':
        cinic_subdir = 'train'
    elif split == 'val':
        cinic_subdir = 'valid'
    elif split == 'test':
        cinic_subdir = 'test'
    else:
        raise ValueError()

    cinic_directory = os.path.join(path, cinic_subdir)
    cinic_dataset = ImageFolder(cinic_directory,transform=transform)

    loader = torch.utils.data.DataLoader(cinic_dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'CINIC-10'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader
