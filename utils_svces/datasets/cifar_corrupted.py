import torch
import torch.distributions
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import os
import numpy as np

from .paths import get_CIFAR10_C_path, get_CIFAR100_C_path
from utils_svces.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128

#cheers to tf: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/cifar10_corrupted.py
_CORRUPTIONS_TO_FILENAMES = {
    'gaussian_noise': 'gaussian_noise.npy',
    'shot_noise': 'shot_noise.npy',
    'impulse_noise': 'impulse_noise.npy',
    'defocus_blur': 'defocus_blur.npy',
    'frosted_glass_blur': 'glass_blur.npy',
    'motion_blur': 'motion_blur.npy',
    'zoom_blur': 'zoom_blur.npy',
    'snow': 'snow.npy',
    'frost': 'frost.npy',
    'fog': 'fog.npy',
    'brightness': 'brightness.npy',
    'contrast': 'contrast.npy',
    'elastic': 'elastic_transform.npy',
    'pixelate': 'pixelate.npy',
    'jpeg_compression': 'jpeg_compression.npy',
    'gaussian_blur': 'gaussian_blur.npy',
    'saturate': 'saturate.npy',
    'spatter': 'spatter.npy',
    'speckle_noise': 'speckle_noise.npy',
}

_LABELS_FILENAME = 'labels.npy'

BENCHMARK_CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'frosted_glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic',
    'pixelate',
    'jpeg_compression',
]

EXTRA_CORRUPTIONS = [
    'gaussian_blur',
    'saturate',
    'spatter',
    'speckle_noise',
]

def get_CIFAR10_C(split='benchmark', severity=1, batch_size=None, shuffle=False,
                  augm_type='none', cutout_window=16, num_workers=2, size=32, config_dict=None):
    if batch_size == None:
        batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window, out_size=size, config_dict=augm_config)

    path = get_CIFAR10_C_path()
    dataset = CIFARCorrupted(path, split=split, severity=severity, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Cifar10-C'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader

def get_CIFAR100_C(split='benchmark', severity=1, batch_size=None, shuffle=False,
                   augm_type='none', cutout_window=8, num_workers=2, size=32, config_dict=None):
    if batch_size == None:
        batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window, out_size=size, config_dict=augm_config)

    path = get_CIFAR100_C_path()
    dataset = CIFARCorrupted(path, split=split, severity=severity, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Cifar100-C'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader

class CIFARCorrupted(Dataset):
    def __init__(self, path, split='benchmark', severity=1, transform=None):
        self.transform = transform

        if split == 'benchmark':
            corruptions = BENCHMARK_CORRUPTIONS
        elif split == 'extra':
            corruptions = EXTRA_CORRUPTIONS
        else:
            raise NotImplementedError()

        assert ((severity >= 1) & (severity <= 5))

        labels_file = os.path.join(path, _LABELS_FILENAME)
        labels = np.load(labels_file)
        num_images = labels.shape[0] // 5
        # Labels are stacked 5 times so we can just read the first iteration
        self.labels = labels[:num_images]

        total_images = len(corruptions) * num_images
        images = np.zeros((total_images, 32, 32, 3), dtype=np.uint8)

        severity_idx_start = (severity - 1) * num_images
        severity_idx_end = (severity) * num_images

        for i, corruption in enumerate(corruptions):
            idx_start = i * num_images
            idx_end = (i+1) * num_images

            images_i_filename = os.path.join(path, _CORRUPTIONS_TO_FILENAMES[corruption])
            images_i =np.load(images_i_filename)[severity_idx_start:severity_idx_end]

            images[idx_start:idx_end] = images_i

        self.images = images
        self.length = total_images
        self.images_per_corruption = num_images

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label_index = index % self.images_per_corruption
        label = self.labels[label_index]
        return img, label

    def __len__(self):
        return self.length
