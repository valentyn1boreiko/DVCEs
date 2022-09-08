import torch
import torch.distributions
from torchvision import datasets
from torch.utils.data import DataLoader

from .paths import get_imagenet_o_path
from utils_svces.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128

def get_imagenet_o(batch_size=None, shuffle=False, augm_type='none',
                           num_workers=8, size=224):
    if batch_size == None:
        batch_size = DEFAULT_TEST_BATCHSIZE

    transform = get_imageNet_augmentation(type=augm_type, out_size=size)

    path = get_imagenet_o_path()

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)
    return loader
