import torch
import torch.distributions
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from .preproc import PermutationNoise, GaussianFilter, ContrastRescaling
from .paths import get_base_data_dir, get_svhn_path, get_CIFAR100_path, get_CIFAR10_path
import numpy as np


DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128

def get_permutationNoise(dataset, train=True, batch_size=None):
    if batch_size==None:
        if train:
            batch_size=DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size=DEFAULT_TEST_BATCHSIZE
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    PermutationNoise(),
                    GaussianFilter(),
                    ContrastRescaling()
                    ])

    path = get_base_data_dir()
    if dataset=='MNIST':
        dataset = datasets.MNIST(path, train=train, transform=transform)
    elif dataset=='FMNIST':
        dataset = datasets.FashionMNIST(path, train=train, transform=transform)
    elif dataset=='SVHN':
        dataset = datasets.SVHN(get_svhn_path(), split='train' if train else 'test', transform=transform)
    elif dataset=='CIFAR10':
        dataset = datasets.CIFAR10(get_CIFAR10_path(), train=train, transform=transform)
    elif dataset=='CIFAR100':
        dataset = datasets.CIFAR100(get_CIFAR100_path(), train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    #cifar_loader = PrecomputeLoader(cifar_loader, batch_size=batch_size, shuffle=True)
    return loader


def get_UniformNoise(dataset, train=False, batch_size=None):
    if batch_size==None:
        if train:
            batch_size=DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size=DEFAULT_TEST_BATCHSIZE
    import torch.utils.data as data_utils

    if dataset in ['MNIST', 'FMNIST']:
        shape = (1, 28, 28)
    elif dataset in ['SVHN', 'CIFAR10', 'CIFAR100']:
        shape = (3, 32, 32)
    elif dataset in ['imageNet', 'restrictedImageNet']:
        shape = (3, 224, 224)

    data = torch.rand((100*batch_size,) + shape)
    train = data_utils.TensorDataset(data, torch.zeros(data.shape[0], device=data.device))
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    return loader

class UniormNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, dim, length=100000000):

        def load_image(idx):
            return torch.rand(dim)

        self.load_image = load_image

        self.length = length

        transform = None

        self.transform = transform

    def __getitem__(self, index):

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return self.length


def ImageNetMinusCifar10(train=False, batch_size=None, augm_flag=False):
    if train:
        print('Warning: Training set for ImageNet not available')
    if batch_size is None:
        batch_size = DEFAULT_TEST_BATCHSIZE

    path = get_base_data_dir()
    dir_imagenet = path + '/imagenet/val/'
    n_test_imagenet = 30000

    transform = transforms.ToTensor()

    dataset = torch.utils.data.Subset(datasets.ImageFolder(dir_imagenet, transform=transform),
                                            np.random.permutation(range(n_test_imagenet))[:10000])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    return loader
