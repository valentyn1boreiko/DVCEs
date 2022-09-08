import torch
import torch.distributions
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os
from torchvision.datasets.folder import default_loader
from scipy.io import loadmat

from .paths import get_flowers_path
from utils_svces.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation

FLOWERS_LABELS = [
                    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
                    "sweet pea", "english marigold", "tiger lily", "moon orchid",
                    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
                    "colt's foot", "king protea", "spear thistle", "yellow iris",
                    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
                    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
                    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
                    "stemless gentian", "artichoke", "sweet william", "carnation",
                    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
                    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
                    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
                    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
                    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
                    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
                    "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
                    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
                    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
                    "azalea", "water lily", "rose", "thorn apple", "morning glory",
                    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
                    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
                    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
                    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
                    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
                    "blackberry lily"
                  ]

def get_flowers_labels():
    return FLOWERS_LABELS

def get_flowers(split='train', batch_size=128, shuffle=True, augm_type='none',
                      size=224, num_workers=8, config_dict=None):

    augm_config = {}
    transform = get_imageNet_augmentation(augm_type, out_size=size, config_dict=augm_config)
    path = get_flowers_path()
    dataset = Flowers(path, split, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'Flowers'
        config_dict['Batch out_size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader


class Flowers(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.labels = loadmat(os.path.join(root, 'imagelabels.mat'))['labels'][0].astype(np.long)
        self.transform = transform
        setids = loadmat(os.path.join(root, 'setid.mat'))

        if split == 'train':
            self.indices = setids['trnid'][0]
        elif split =='val':
            self.indices = setids['valid'][0]
        elif split =='train_val':
            trn_idcs = setids['trnid'][0]
            val_idcs = setids['valid'][0]
            self.indices = np.concatenate([trn_idcs, val_idcs])
        elif split == 'test':
            self.indices = setids['tstid'][0]
        else:
            raise ValueError()

        self.indices = self.indices
        self.loader = default_loader
        self.length = len(self.indices)

    def __getitem__(self, index):
        img_idx = self.indices[index]
        #matlab starts with 1, so decrease both index and target idx by 1
        target = self.labels[img_idx - 1] - 1
        path = os.path.join(self.root, 'jpg', f'image_{img_idx:05d}.jpg')
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length
