import torch
import torch.distributions
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os

from utils_svces.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation
from .paths import get_tiny_images_files

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128

def get_80MTinyImages(batch_size=100, augm_type='default', shuffle=True, cutout_window=16, num_workers=1,
                      size=32, exclude_cifar=False, exclude_cifar10_1=False, config_dict=None):
    #dataset is the dataset that will be excluded, eg CIFAR10
    if num_workers > 1:
        pass
        #raise ValueError('Bug in the current multithreaded tinyimages implementation')

    augm_config = {}
    transform = get_cifar10_augmentation(augm_type, cutout_window=cutout_window, out_size=size, config_dict=augm_config)

    dataset_out = TinyImagesDataset(transform,
                                    exclude_cifar=exclude_cifar, exclude_cifar10_1=exclude_cifar10_1)

    loader = torch.utils.data.DataLoader(dataset_out, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        if config_dict is not None:
            config_dict['Dataset'] = '80M Tiny Images'
            config_dict['Shuffle'] = shuffle
            config_dict['Batch out_size'] = batch_size
            config_dict['Exclude CIFAR'] = exclude_cifar
            config_dict['Exclude CIFAR10.1'] = exclude_cifar10_1
            config_dict['Augmentation'] = augm_config

    return loader

def _preload_tiny_images(idcs, file_id):
    imgs = np.zeros((len(idcs), 32, 32, 3), dtype='uint8')
    for lin_idx, idx in enumerate(idcs):
        imgs[lin_idx,:] = _load_tiny_image(idx, file_id)
    return imgs

def _load_tiny_image(idx, file_id):
    try:
        file_id.seek(idx * 3072)
        data = file_id.read(3072)
    finally:
        pass

    data_np = np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")
    return data_np


def _load_cifar_exclusion_idcs(exclude_cifar, exclude_cifar10_1):
    cifar_idxs = []
    main_idcs_dir = 'TinyImagesExclusionIdcs/'

    our_exclusion_files = [
                        '80mn_cifar10_test_idxs.txt',
                        '80mn_cifar100_test_idxs.txt',
                        '80mn_cifar10_train_idxs.txt',
                        '80mn_cifar100_train_idxs.txt',
                      ]
    if exclude_cifar:
        with open(os.path.join(main_idcs_dir, '80mn_cifar_idxs.txt'), 'r') as idxs:
            for idx in idxs:
                # indices in file take the 80mn database to start at 1, hence "- 1"
                cifar_idxs.append(int(idx) - 1)

        for file in our_exclusion_files:
            with open(os.path.join(main_idcs_dir, file), 'r') as idxs:
                for idx in idxs:
                    cifar_idxs.append(int(idx))

    if exclude_cifar10_1:
        with open(os.path.join(main_idcs_dir, '80mn_cifar101_idxs.txt'), 'r') as idxs:
            for idx in idxs:
                cifar_idxs.append(int(idx))

    cifar_idxs = torch.unique(torch.LongTensor(cifar_idxs))
    return cifar_idxs

TINY_LENGTH = 79302017

# Code from https://github.com/hendrycks/outlier-exposure
class TinyImagesDataset(Dataset):
    def __init__(self, transform_base, exclude_cifar=False, exclude_cifar10_1=False):
        self.data_location = get_tiny_images_files(False)
        self.memap = np.memmap(self.data_location, mode='r', dtype='uint8', order='C').reshape(TINY_LENGTH, -1)

        if transform_base is not None:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transform_base])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()])

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        exclusion_idcs = _load_cifar_exclusion_idcs(exclude_cifar, exclude_cifar10_1)

        self.included_indices = torch.ones(TINY_LENGTH, dtype=torch.long)
        self.included_indices[exclusion_idcs] = 0
        self.included_indices = torch.nonzero(self.included_indices, as_tuple=False).squeeze()
        self.length = len(self.included_indices)
        print(f'80M Tiny Images - Length {self.length} - Excluding {len(exclusion_idcs)} images')

    def __getitem__(self, ii):
        index = self.included_indices[ii]
        img = self.memap[index].reshape(32, 32, 3, order="F")

        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return self.length



