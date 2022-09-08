import torch
from torchvision import transforms

from .autoaugment import CIFAR10Policy, ImageNetPolicy
from .cutout import Cutout
from .utils import INTERPOLATION_STRING_TO_TYPE

CIFAR10_mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)

DEFAULT_CIFAR10_PARAMETERS = {
    'interpolation': 'bilinear',
    'mean': CIFAR10_mean,
    'crop_pct': 0.875
}

def get_cifar10_augmentation(type='default', cutout_window=16, out_size=32, in_size=32, magnitude_factor=1,
                             augm_parameters=None, config_dict=None):
    if augm_parameters is None:
        augm_parameters = DEFAULT_CIFAR10_PARAMETERS

    cutout_color = torch.tensor([0., 0., 0.])
    mean_int = tuple(int(255. * v) for v in augm_parameters['mean'])
    mean_tensor = torch.FloatTensor(augm_parameters['mean'])
    padding_size = int((1. - augm_parameters['crop_pct']) * in_size)
    interpolation_mode = INTERPOLATION_STRING_TO_TYPE[augm_parameters['interpolation']]
    force_no_resize = False

    if type == 'none' or type is None:
        transform_list = []
    elif type == 'default' or type == 'default_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=mean_int),
            transforms.RandomHorizontalFlip(),
        ]
        cutout_color = mean_tensor
    elif 'jitter' in type:
        #jitter strength
        s = float(type[7:])
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=mean_int),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
            transforms.RandomGrayscale(p=0.2),
        ]
        cutout_color = mean_tensor
    elif type == 'madry' or type == 'madry_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=mean_int),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
        ]
        cutout_color = mean_tensor
    elif type == 'autoaugment' or type == 'autoaugment_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=mean_int),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(fillcolor=mean_int, magnitude_factor=magnitude_factor),
        ]
        cutout_color = mean_tensor
    elif type == 'in_autoaugment' or type == 'in_autoaugment_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=mean_int),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(fillcolor=mean_int),
        ]
        cutout_color = mean_tensor
    elif type == 'big_transfer' or type == 'big_transfer_128':
        if type == 'big_transfer':
            if out_size != 480:
                print(f'Out out_size of {out_size} detected but Big Transfer is supposed to be used with 480')
                pre_crop_size = int(out_size * (512/480))
            else:
                pre_crop_size = 512
        else:
            if out_size != 128:
                print(f'Out out_size of {out_size} detected but Big Transfer 128 is supposed to be used with 128')
                pre_crop_size = int(out_size * (160 / 128))
            else:
                pre_crop_size = 160

        print(f'BigTransfer augmentation: Pre crop {pre_crop_size} - Out Size {out_size}')
        transform_list = [
            transforms.transforms.Resize((pre_crop_size, pre_crop_size), interpolation=interpolation_mode),
            transforms.transforms.RandomCrop((out_size, out_size)),
            transforms.transforms.RandomHorizontalFlip(),
        ]
        force_no_resize = True
    else:
        raise ValueError(f'augmentation type - {type} - not supported')

    if out_size != in_size and not force_no_resize:
        if 'cutout' in type:
            transform_list.append(transforms.Resize(out_size, interpolation=interpolation_mode))
            transform_list.append(transforms.ToTensor())
            cutout_size = int(out_size / in_size * cutout_window)
            print(f'Relative Cutout window {cutout_window / in_size} - Absolute Cutout window: {cutout_size}')
            transform_list.append(Cutout(n_holes=1, length=cutout_size, fill_color=cutout_color))
        else:
            transform_list.append(transforms.Resize(out_size, interpolation=interpolation_mode))
            transform_list.append(transforms.ToTensor())
    elif 'cutout' in type:
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
        config_dict['Magnitude factor'] = magnitude_factor
        if 'cutout' in type:
            config_dict['Cutout out_size'] = cutout_size
        for key, value in augm_parameters.items():
            config_dict[key] = value

    return transform