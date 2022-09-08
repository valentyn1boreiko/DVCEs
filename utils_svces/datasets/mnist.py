import torch
import torch.distributions
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 512

def MNIST(train=True, batch_size=None, augm_flag=True, shuffle=None):
    if batch_size==None:
        if train:
            batch_size=DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size=DEFAULT_TEST_BATCHSIZE

    if shuffle is None:
        shuffle = train

    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.MNIST(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=4)
    return loader

def EMNIST(train=False, batch_size=None, augm_flag=False, shuffle=None):
    if batch_size==None:
        if train:
            batch_size=DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size=DEFAULT_TEST_BATCHSIZE

    if shuffle is None:
        shuffle = train

    transform_base = [transforms.ToTensor(), pre.Transpose()] #EMNIST is rotated 90 degrees from MNIST
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.EMNIST(path, split='letters',
                              train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=1)
    return loader


def FMNIST(train=False, batch_size=None, augm_flag=False, shuffle=None):
    if batch_size==None:
        if train:
            batch_size=DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size=DEFAULT_TEST_BATCHSIZE

    if shuffle is None:
        shuffle = train

    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.FashionMNIST(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=1)
    return loader
