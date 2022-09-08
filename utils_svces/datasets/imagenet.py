import torch
import torch.distributions
from torchvision import datasets
from torch.utils.data import DataLoader
from utils_svces.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation
from .paths import get_imagenet_path

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128


def get_imagenet_labels(path_base=None):
    path = get_imagenet_path(path_base)
    dataset = datasets.ImageNet(path, split='val', transform='none')
    classes_extended = dataset.classes
    labels = []
    for a in classes_extended:
        labels.append(a[0])
    return labels

def get_imagenet_label_wid_pairs():
    path = get_imagenet_path()
    dataset = datasets.ImageNet(path, split='val', transform='none')
    classes_extended = dataset.classes
    wids = dataset.wnids

    label_wid_pairs = []
    for a, b in zip(classes_extended, wids) :
        label_wid_pairs.append((a[0], b))
    return label_wid_pairs

def get_ImageNet(path='', train=True, batch_size=None, shuffle=None, augm_type='test', num_workers=8, size=224, config_dict=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_imageNet_augmentation(type=augm_type, out_size=size, config_dict=augm_config)
    if not train and augm_type != 'none':
        print('Warning: ImageNet test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path += '/imagenet'

    if train == True:
        dataset = datasets.ImageNet(path, split='train', transform=transform)
    else:
        dataset = datasets.ImageNet(path, split='val', transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'ImageNet'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader






