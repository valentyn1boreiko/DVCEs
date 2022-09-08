import os

import torch
import torch.distributions
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader

from utils_svces.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation
from .paths import get_pets_path

class_labels = ['Abyssinian', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle', 'Bengal',
                'Birman', 'Bombay', 'boxer', 'British Shorthair', 'chihuahua', 'Egyptian Mau', 'english cocker spaniel',
                'english setter', 'german shorthaired', 'great pyrenees', 'havanese', 'japanese chin', 'keeshond',
                'leonberger', 'Maine Coon', 'miniature pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug',
                'Ragdoll', 'Russian Blue', 'saint bernard', 'samoyed', 'scottish terrier', 'shiba inu', 'Siamese',
                'Sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier']

def get_pets_labels():
    return class_labels

def get_pets(split='train', batch_size=128, shuffle=True, augm_type='none',
                      size=224, num_workers=8, config_dict=None):

    augm_config = {}
    transform = get_imageNet_augmentation(augm_type, out_size=size, config_dict=augm_config)
    path = get_pets_path()
    dataset = Pets(path, split, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Flowers'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader


class Pets(Dataset):
    def __init__(self, root, split, transform=None):
        if split == 'train':
            annotations_file = os.path.join(root, 'annotations/trainval.txt')
        elif split == 'test':
            annotations_file = os.path.join(root, 'annotations/test.txt')
        else:
            raise ValueError(f'Split {split} not supported')

        self.img_root = os.path.join(root, 'images')

        self.labels = []
        self.imgs = []

        with open(annotations_file, 'r') as fileID:
            for line in fileID:
                line_parts = line.rstrip().split(' ')
                img = line_parts[0]
                label = int(line_parts[1]) - 1 #labels are in range 1:37, transform to 0:36

                self.imgs.append(img)
                self.labels.append(label)

        self.transform = transform
        self.loader = default_loader
        self.length = len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        target = self.labels[index]
        path = os.path.join(self.img_root, f'{img}.jpg')
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length
