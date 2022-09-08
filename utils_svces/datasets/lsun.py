import torch
import torch.distributions
from torchvision import datasets, transforms

from .paths import get_base_data_dir, get_LSUN_scenes_path
from torch.utils.data import DataLoader, SubsetRandomSampler

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128

from utils_svces.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation
from utils_svces.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation

# LSUN classroom
def get_LSUN_CR(train=False, batch_size=None, size=32):
    if train:
        ValueError('Warning: Training set for LSUN not available')
    if batch_size is None:
        batch_size=DEFAULT_TEST_BATCHSIZE

    transform = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.ToTensor()
        ])
    path = get_base_data_dir()
    data_dir = path + '/LSUN'
    dataset = datasets.LSUN(data_dir, classes=['classroom_val'], transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    return loader

def get_LSUN_scenes(split='train', samples_per_class=None, batch_size=None, shuffle=None, augm_type='none',
                    augm_class='imagenet', num_workers=8, size=224, config_dict=None):
    if batch_size is None:
        batch_size=DEFAULT_TEST_BATCHSIZE

    augm_config = {}

    if augm_class == 'imagenet':
        transform = get_imageNet_augmentation(type=augm_type, out_size=size, config_dict=augm_config)
    elif augm_class == 'cifar':
        raise NotImplementedError()
        transform = get_cifar10_augmentation(type=augm_type, out_size=size, in_size=224, config_dict=augm_config)
    else:
        raise NotImplementedError()
    path = get_LSUN_scenes_path()
    dataset = datasets.LSUN(path, classes=split, transform=transform)

    if samples_per_class is None:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

    else:
        num_classes = len(dataset.dbs)
        idcs = torch.zeros(num_classes, samples_per_class, dtype=torch.long)
        start_idx = 0
        for i in range(num_classes):
            idcs[i, :] = torch.arange(start_idx,start_idx + samples_per_class)
            start_idx = dataset.indices[i]
        idcs = idcs.view(-1).numpy()
        sampler = SubsetRandomSampler(idcs)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers)

    return loader


def get_LSUN_scenes_labels():
    return  ['bedroom', 'bridge', 'church_outdoor', 'classroom',
             'conference_room', 'dining_room', 'kitchen',
             'living_room', 'restaurant', 'tower']
