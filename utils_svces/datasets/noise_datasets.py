import torch
import torch.distributions
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from utils_svces.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation

DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128


def get_noise_dataset(length, type='normal', batch_size=128, augm_type='none', cutout_window=32,
                      num_workers=8, size=32, config_dict=None):
    augm_config = {}
    transform = get_cifar10_augmentation(type=augm_type, cutout_window=cutout_window, out_size=size,
                                         in_size=size, config_dict=augm_config)

    dataset = NoiseDataset(length, type, size, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'NoiseData'
        config_dict['Length'] = length
        config_dict['Noise Type'] = type
        config_dict['Batch size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader

class NoiseDataset(Dataset):
    def __init__(self, length, type, size, transform):
        assert type in ['uniform', 'normal']

        self.type = type
        self.size = size
        self.length = length

        np.random.seed(123)
        if self.type == 'uniform':
            data_np = np.random.rand(length, 3, self.size, self.size).astype(np.float32)
            self.data = torch.from_numpy(data_np)
        elif self.type == 'normal':
            data_np = 0.5 + np.random.randn(length, 3, self.size, self.size).astype(np.float32)
            self.data = torch.clamp(torch.from_numpy(data_np), min=0, max=1)
        else:
            raise NotImplementedError()


        transform = transforms.Compose([
            transforms.ToPILImage(),
            transform])

        self.transform = transform

    def __getitem__(self, index):
        target = 0
        img = self.data[index].squeeze(dim=0)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length
