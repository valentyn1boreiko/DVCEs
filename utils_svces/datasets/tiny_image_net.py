import os
import torch
import torch.distributions
from .paths import get_tiny_imagenet_path
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from utils_svces.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128


def get_TinyImageNetClassNames(cleaned=True):
    class_labels = []
    path = get_tiny_imagenet_path()
    with open(f'{path}label_clearnames.txt', 'r') as fileID:
        for line_idx, line in enumerate(fileID.readlines()):
            line_elements = str(line).split("\t")
            class_labels.append(line_elements[1].rstrip())


    if cleaned:
        class_labels_cleaned = []
        for label in class_labels:
            class_labels_cleaned.append(label.split(',')[0])
    else:
        class_labels_cleaned = class_labels

    return class_labels_cleaned


def get_TinyImageNet(split, batch_size=None, shuffle=None, augm_type='none', cutout_window=32, num_workers=8, size=64,  config_dict=None):
    if batch_size == None:
        if split == 'train':
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window, out_size=size,
                                         in_size=64, config_dict=augm_config)

    if shuffle is None:
        shuffle = True if split == 'train' else False


    path = get_tiny_imagenet_path()
    dataset = TinyImageNet(path, split, transform_base=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'TinyImageNet'
        config_dict['Batch size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader

class TinyImageNet(ImageFolder):
    def __init__(self, path, split, transform_base):
        assert split in ['train', 'test', 'val']

        root = os.path.join(path, split)
        super().__init__(root, transform=transform_base)

        print(f'TinyImageNet {split} - Length {len(self)}')
