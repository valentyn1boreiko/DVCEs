import torch
import torch.distributions
from torchvision import datasets, transforms
from .paths import get_celebA_path, get_imagenet_path
from utils_svces.datasets.augmentations.cutout import Cutout

def get_celebA_augmentation(type='default', out_size=224, config_dict=None):
    celebA_mean = torch.tensor([0.5063, 0.4258, 0.3832])
    celebA_mean_int = (int(255 * 0.5063), int(255 * 0.4258), int(255 * 0.3832))

    if type == 'none' or type is None:
        transform = transforms.Compose([
            transforms.Resize(out_size),
            transforms.ToTensor()])
    elif type == 'default':
        transform = transforms.Compose([
            transforms.Resize(out_size),
            transforms.RandomCrop(out_size, padding=int(0.125 * out_size), fill=celebA_mean_int),
            transforms.ToTensor(),
        ])
    elif type == 'madry':
        transform = transforms.Compose([
            transforms.Resize(out_size),
            transforms.RandomCrop(out_size, padding=int(0.125 * out_size), fill=celebA_mean_int),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
    elif type == 'default_cutout':
        transform = transforms.Compose([
            transforms.Resize(out_size),
            transforms.RandomCrop(out_size, padding=int(0.125 * out_size), fill=celebA_mean_int),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #as we apply cutout before normalization, we have to fill with the mean and not 0
            Cutout(n_holes=1, length=int(0.25 * out_size), fill_color= celebA_mean)
        ])
    elif type == 'madry_cutout':
        transform = transforms.Compose([
            transforms.Resize(out_size),
            transforms.RandomCrop(out_size, padding=int(0.125 * out_size), fill=celebA_mean_int),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=int(0.4 * out_size), fill_color= celebA_mean)
        ])
    else:
        raise ValueError(f'augmentation type - {type} - not supported')

    if config_dict is not None:
        config_dict['type'] = type
        config_dict['Output out_size'] = out_size

    return transform


celebA_attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                     'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                     'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                     'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                     'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                     'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                     'Wearing_Necktie', 'Young'
                    ]

def celebA_target_transform(targets, mask):
    return targets[mask]

def get_celebA_target_transform(attributes):
    mask = []
    for i, attr in enumerate(celebA_attributes):
        if attr in attributes:
            mask.append(i)

    mask = torch.LongTensor(mask)
    transform = lambda x: celebA_target_transform(x, mask)
    return transform


def celebA_feature_set(split='train', shuffle=None, batch_size=128, attributes=None,
                       augm_type='default', out_size=224, config_dict=None):
    if split == 'test' and not augm_type == 'none':
        print('WARNING: Test set in use with data augmentation')

    if shuffle is None:
        if split == 'train':
            shuffle = True
        else:
            shuffle = False

    if attributes is None:
        attributes = celebA_attributes
        target_transform = None
    else:
        target_transform = get_celebA_target_transform(attributes)

    augm_config = {}
    augm = get_celebA_augmentation(augm_type, out_size=out_size, config_dict=augm_config)

    path = get_celebA_path()
    dataset = datasets.CelebA(path, split=split, target_type='attr', transform=augm, target_transform=target_transform, download=False )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=8)

    if config_dict is not None:
        config_dict['Dataset'] = 'CelebA'
        config_dict['Batch size'] = batch_size
        config_dict['Augmentation'] = augm_config
        config_dict['Attributes'] = attributes

    return loader

def celebA_ImageNetOD(shuffle=True, batch_size=128, augm_type='default'):
    augm = get_celebA_augmentation(augm_type)
    root = get_imagenet_path()
    dataset = datasets.ImageNet(root=root, split='train', transform=augm)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=8)
    return loader
