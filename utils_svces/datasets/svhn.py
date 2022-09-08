import torch
import torch.distributions
from torchvision import datasets
from torch.utils.data import Dataset

from .combo_dataset import ComboDataset
from .paths import get_svhn_path
from utils_svces.datasets.augmentations.svhn_augmentation import get_SVHN_augmentation

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128


def get_SVHN_labels():
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return class_labels

class SVHNTrainExtraCombo(ComboDataset):
    def __init__(self, transform=None):
        path = get_svhn_path()
        train = datasets.SVHN(path, split='train', transform=transform, download=True)
        extra = datasets.SVHN(path, split='extra', transform=transform, download=True)

        super().__init__([train, extra])
        print(f'SVHN Train + Extra - Train: {len(train)} - Extra {len(extra)} - Total {self.length}')

def get_SVHN(split='train', shuffle = None, batch_size=None, augm_type='none', size=32, num_workers=4, config_dict=None):
    if batch_size==None:
        if split in ['train', 'extra']:
            batch_size=DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size=DEFAULT_TEST_BATCHSIZE

    if shuffle is None:
        if split in ['train', 'extra']:
            shuffle = True
        else:
            shuffle = False

    augm_config = {}
    transform = get_SVHN_augmentation(augm_type, out_size=size, config_dict=augm_config)

    path = get_svhn_path()
    if split=='svhn_train_extra':
        dataset = SVHNTrainExtraCombo(transform)
    else:
        dataset = datasets.SVHN(path, split=split, transform=transform, download=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'SVHN'
        config_dict['SVHN Split'] = split
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader
