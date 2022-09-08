import torch
import torch.distributions
from torch.utils.data import DataLoader
import os

from .paths import get_food_101N_path
from utils_svces.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation
from torchvision.datasets.folder import default_loader
import csv


def get_food_101N_labels():
    path = get_food_101N_path()
    class_list = []
    classes_file = os.path.join(path, 'meta', 'classes.txt')
    with open(classes_file) as classestxt:
        for line_number, line in enumerate(classestxt):
            if line_number > 0: #skip the first line as it's no class
                class_list.append(line.rstrip())
    return class_list


def get_food_101N(split='train', batch_size=128, shuffle=True, augm_type='none',
                      size=224, num_workers=8):
    transform = get_imageNet_augmentation(augm_type, out_size=size)
    path = get_food_101N_path()
    dataset = Food101N(path, split, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)
    return loader


class Food101N(torch.utils.data.Dataset):
    def __init__(self, root, split, verified_only=True, transform=None):
        class_labels = get_food_101N_labels()
        self.root = root
        label_to_target = {label:target for target, label in enumerate(class_labels)}
        self.transform = transform

        if split == 'train':
            meta_tsv = os.path.join(self.root, 'meta', 'verified_train.tsv')
        elif split == 'val':
            meta_tsv = os.path.join(self.root, 'meta', 'verified_val.tsv')
        else:
            raise ValueError()

        self.img_label_list = []
        with open(meta_tsv)as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab')
            for row in reader:
                if verified_only and row['verification_label']:
                    img = row['class_name/key']
                    target = label_to_target[img.split('/')[0]]
                    self.img_label_list.append((img,target))

        print(f'Food 101N {split} - Verified only {verified_only} - {len(self.img_label_list)} Images')

        self.loader = default_loader
        self.length = len(self.img_label_list)

    def __getitem__(self, index):
        sub_path, target = self.img_label_list[index]
        path = os.path.join(self.root, 'images', sub_path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length
