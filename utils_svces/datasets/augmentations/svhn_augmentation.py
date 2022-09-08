from torchvision import transforms
import torch
from utils_svces.datasets.augmentations.autoaugment import SVHNPolicy, CIFAR10Policy
from utils_svces.datasets.augmentations.cutout import Cutout
from .utils import INTERPOLATION_STRING_TO_TYPE

SVHN_mean = (0.4377, 0.4438, 0.4728)

DEFAULT_SVHN_PARAMETERS = {
    'interpolation': 'bilinear',
    'mean': SVHN_mean,
    'crop_pct': 0.875
}

def get_SVHN_augmentation(augm_type='none', in_size=32, out_size=32, augm_parameters=None, config_dict=None):
    if augm_parameters is None:
        augm_parameters = DEFAULT_SVHN_PARAMETERS

    mean_int = tuple(int(255. * v) for v in augm_parameters['mean'])
    mean_tensor = torch.FloatTensor(augm_parameters['mean'])
    padding_size = int((1. - augm_parameters['crop_pct']) * in_size)
    interpolation_mode = INTERPOLATION_STRING_TO_TYPE[augm_parameters['interpolation']]
    
    if augm_type == 'none':
        transform_list = []
    elif augm_type == 'default' or augm_type == 'default_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=mean_int),
        ]
    elif augm_type == 'autoaugment' or augm_type == 'autoaugment_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=mean_int),
            SVHNPolicy(fillcolor=mean_int),
        ]
    elif augm_type == 'cifar_autoaugment' or augm_type == 'cifar_autoaugment_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=mean_int),
            CIFAR10Policy(fillcolor=mean_int),
        ]
    else:
        raise ValueError()

    cutout_window = 16
    cutout_color = mean_tensor
    cutout_size = 0

    if out_size != in_size:
        if 'cutout' in augm_type:
            transform_list.append(transforms.Resize(out_size, interpolation=interpolation_mode))
            transform_list.append(transforms.ToTensor())
            cutout_size = int(out_size / in_size * cutout_window)
            print(f'Relative Cutout window {cutout_window / in_size} - Absolute Cutout window: {cutout_size}')
            transform_list.append(Cutout(n_holes=1, length=cutout_size, fill_color=cutout_color))
        else:
            transform_list.append(transforms.Resize(out_size, interpolation=interpolation_mode))
            transform_list.append(transforms.ToTensor())
    elif 'cutout' in augm_type:
        cutout_size = cutout_window
        print(f'Relative Cutout window {cutout_size / in_size} - Absolute Cutout window: {cutout_size}')
        transform_list.append(transforms.ToTensor())
        transform_list.append(Cutout(n_holes=1, length=cutout_size, fill_color=cutout_color))
    else:
        transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    if config_dict is not None:
        config_dict['type'] = type
        config_dict['Input size'] = in_size
        config_dict['Output size'] = out_size
        if 'cutout' in augm_type:
            config_dict['Cutout out_size'] = cutout_size
        for key, value in augm_parameters.items():
            config_dict[key] = value

    return transform