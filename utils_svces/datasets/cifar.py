import torch
import torch.distributions
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader, Dataset

from .augmentations.cifar_augmentation import get_cifar10_augmentation
from .preproc import Gray
import numpy as np
#from .auto_augmen_old import AutoAugment
from .paths import get_CIFAR10_path, get_CIFAR100_path, get_base_data_dir
from PIL import Image
import pickle
import os

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128

def GrayCIFAR10(train=False, batch_size=None, augm_flag=False, shuffle=None, resolution=28):
    if batch_size==None:
        if train:
            batch_size=DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size=DEFAULT_TEST_BATCHSIZE
    transform_base = [transforms.Compose([
                            transforms.Resize(resolution),
                            transforms.ToTensor(),
                            Gray()
                       ])]
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(resolution, padding=4, padding_mode='reflect'),
        ] + transform_base)

    if shuffle is None:
        shuffle = train

    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    path = get_CIFAR10_path()
    dataset = datasets.CIFAR10(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=1)
    return loader


def get_CIFAR10(train=True, batch_size=None, shuffle=None, augm_type='none', cutout_window=16, num_workers=2, size=32, config_dict=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window, out_size=size, config_dict=augm_config)
    if not train and augm_type != 'none':
        print('Warning: CIFAR10 test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_CIFAR10_path()
    dataset = datasets.CIFAR10(path, train=train, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Cifar10'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader

def get_CIFAR10_labels():
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return labels


def get_CIFAR100(train=True, batch_size=None, shuffle=None, augm_type='none', cutout_window=8, num_workers=2, size=32, config_dict=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window, out_size=size, config_dict=augm_config)
    if not train and augm_type != 'none':
        print('Warning: CIFAR100 test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_CIFAR100_path()
    dataset = datasets.CIFAR100(path, train=train, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Cifar100'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader


def get_CIFAR100_labels():
    path = get_CIFAR100_path()
    infile = open(f'{path}/cifar-100-python/meta', 'rb')
    dict = pickle.load(infile)
    infile.close()
    labels = dict['fine_label_names']
    return labels

class CIFAR10_1Dataset(VisionDataset):
    def __init__(self, transform=None, target_transform=None,
                 version='v6'):

        root = get_base_data_dir()
        super().__init__(root, transform=transform,
                                      target_transform=target_transform)

        subfolder = f'{root}/cifar10_1/'

        if version == 'v4':
            data_file = f'{subfolder}/cifar10.1_v4_data.npy'
            target_file = f'{subfolder}/cifar10.1_v4_labels.npy'
        elif version =='v6':
            data_file = f'{subfolder}/cifar10.1_v6_data.npy'
            target_file = f'{subfolder}/cifar10.1_v6_labels.npy'
        else:
            raise ValueError('Version not supported')

        self.data = np.load(data_file)
        self.targets = np.load(target_file).astype(np.long)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index, :], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



def get_CIFAR10_1(batch_size=None, shuffle=False, augm_type='none', size=32):
    if batch_size == None:
        batch_size = DEFAULT_TEST_BATCHSIZE

    transform = get_cifar10_augmentation(type=augm_type, out_size=size)

    dataset = CIFAR10_1Dataset(transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=8)
    return loader

CIFAR100inCIFAR10Labels = ['pickup_truck', 'bus']
#CIFAR100inCIFAR10Labels = ['pickup_truck']

class CIFAR100MinusCIFAR10(Dataset):
    def __init__(self, path, train=True, samples_per_class=None, transform=None):
        super().__init__()
        self.cifar100 = datasets.CIFAR100(path, train=train, transform=transform)

        self.excluded_idcs = []
        self.classes = self.cifar100.classes.copy()
        for label in CIFAR100inCIFAR10Labels:
            self.classes.remove(label)

        print(f'Cifar100 Minus Cifar10 - Remaining classes: {len(self.classes)}')

        cifar100_targets = torch.LongTensor(self.cifar100.targets)
        cifar100_idcs = []
        targets = []
        for cls_idx, cls in enumerate(self.classes):
            cifar100_class_idx = self.cifar100.class_to_idx[cls]
            class_idcs = torch.nonzero(cifar100_targets == cifar100_class_idx, as_tuple=False).squeeze()
            cifar100_idcs.append(class_idcs)
            targets.append( cls_idx * torch.ones(len(class_idcs), dtype=torch.long))

        self.cifar100_idcs = torch.cat(cifar100_idcs)

        if samples_per_class is None:
            self.subset_idcs = torch.arange(len(self.cifar100_idcs), dtype=torch.long)
            print(f'Using all {len(self.subset_idcs)} samples')
        else:
            split = 'train' if train else 'test'
            idcs_filename = f'cifar100_minus_cifar10_{split}_{samples_per_class}.pt'
            if os.path.exists(idcs_filename):
                self.subset_idcs = torch.load(idcs_filename)
                print(f'Loading an existing subset of {len(self.subset_idcs)} samples')
            else:
                num_samples = samples_per_class * len(cifar100_idcs)
                self.subset_idcs = torch.zeros(num_samples, dtype=torch.long)
                print(f'Creating a subset of {len(self.subset_idcs)} samples')
                offset = 0
                for cls_idx, cls in enumerate(self.classes):
                    num_cls_samples = len(cifar100_idcs[cls_idx])
                    selected_cls_idcs = torch.randperm(num_cls_samples)[:samples_per_class]
                    self.subset_idcs[cls_idx*samples_per_class:(cls_idx+1)*samples_per_class] = offset + selected_cls_idcs
                    offset += num_cls_samples

                torch.save(self.subset_idcs, idcs_filename)

        self.targets = torch.cat(targets)
        self.length = len(self.subset_idcs)

    def __getitem__(self, index):
        sub_idx = self.subset_idcs[index]
        cifar100_idx = self.cifar100_idcs[sub_idx]

        img,_ = self.cifar100[cifar100_idx]
        target = self.targets[sub_idx]

        return img, target

    def __len__(self):
        return self.length



def get_CIFAR100MinusCIFAR10(train=True, batch_size=None, shuffle=None, samples_per_class=None, augm_type='none',
                             cutout_window=16, num_workers=2, size=32, config_dict=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window,
                                         out_size=size, config_dict=augm_config)
    if not train and augm_type != 'none':
        print('Warning: CIFAR10 test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_CIFAR100_path()
    dataset = CIFAR100MinusCIFAR10(path, train=train, samples_per_class=samples_per_class, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Cifar100MinusCifar10'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader

CIFAR10inCIFAR100Labels = ['automobile', 'truck']

class CIFAR10MinusCIFAR100(Dataset):
    def __init__(self, path, train=True, transform=None):
        super().__init__()
        self.cifar10 = datasets.CIFAR10(path, train=train, transform=transform)

        self.excluded_idcs = []
        self.classes = self.cifar10.classes.copy()
        for label in CIFAR10inCIFAR100Labels:
            self.classes.remove(label)

        print(f'Cifar10 Minus Cifar100 - Remaining classes: {len(self.classes)}')

        cifar10_targets = torch.LongTensor(self.cifar10.targets)
        cifar10_idcs = []
        targets = []
        for cls_idx, cls in enumerate(self.classes):
            cifar10_class_idx = self.cifar10.class_to_idx[cls]
            class_idcs = torch.nonzero(cifar10_targets == cifar10_class_idx, as_tuple=False).squeeze()
            cifar10_idcs.append(class_idcs)
            targets.append( cls_idx * torch.ones(len(class_idcs), dtype=torch.long))

        self.cifar10_idcs = torch.cat(cifar10_idcs)
        self.targets = torch.cat(targets)
        self.length = len(self.targets)

    def __getitem__(self, index):
        cifar10_idx = self.cifar10_idcs[index]

        img,_ = self.cifar10[cifar10_idx]
        target = self.targets[index]

        return img, target

    def __len__(self):
        return self.length

def get_CIFAR10MinusCIFAR100(train=True, batch_size=None, shuffle=None, augm_type='none',
                             cutout_window=16, num_workers=2, size=32, config_dict=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window,
                                         out_size=size, config_dict=augm_config)
    if not train and augm_type != 'none':
        print('Warning: CIFAR10 test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_CIFAR10_path()
    dataset = CIFAR10MinusCIFAR100(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Cifar10MinusCifar100'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader
