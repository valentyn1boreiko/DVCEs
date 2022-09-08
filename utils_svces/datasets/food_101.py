import torch
import torch.distributions
from torch.utils.data import DataLoader, Dataset
import os

from .paths import get_food_101_path
from .augmentations.imagenet_augmentation import get_imageNet_augmentation
from torchvision.datasets.folder import default_loader


def get_food_101_labels():
    path = get_food_101_path()
    class_list = []
    classes_file = os.path.join(path, 'meta', 'meta', 'classes.txt')
    with open(classes_file) as classestxt:
        for line_number, line in enumerate(classestxt):
            class_list.append(line.rstrip())
    return class_list


def get_food_101(split='train', batch_size=128, shuffle=True, augm_type='none',
                      size=224, num_workers=8, config_dict=None):

    augm_config = {}
    transform = get_imageNet_augmentation(augm_type, out_size=size, config_dict=augm_config)

    path = get_food_101_path()
    dataset = Food101(path, split, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Food-101'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader

#Adapted to Kaggle Food101 download
#https://www.kaggle.com/kmader/food41
class Food101(Dataset):
    def __init__(self, root, split, transform=None):
        class_labels = get_food_101_labels()
        self.root = root
        label_to_target = {label : target for target, label in enumerate(class_labels)}
        self.transform = transform

        if split == 'train':
            meta_txt = os.path.join(self.root, 'meta', 'meta', 'train.txt')
        elif split == 'val':
            meta_txt = os.path.join(self.root, 'meta', 'meta', 'test.txt')
        else:
            raise ValueError()

        self.img_label_list = []
        with open(meta_txt)as fileID:
            for row in fileID:
                img = row.rstrip()
                target = label_to_target[img.split('/')[0]]
                self.img_label_list.append((img,target))

        print(f'Food 101 {split} - {len(self.img_label_list)} Images')

        self.loader = default_loader
        self.length = len(self.img_label_list)

    def __getitem__(self, index):
        sub_path, target = self.img_label_list[index]
        path = os.path.join(self.root, 'images', sub_path + '.jpg')
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length
