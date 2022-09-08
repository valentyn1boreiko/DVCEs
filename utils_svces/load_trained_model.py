import torch
import os

from utils_svces.models.models_32x32.resnet import ResNet18, ResNet34, ResNet50
from utils_svces.models.models_32x32.wideresnet_carmon import WideResNet as WideResNetCarmon
from utils_svces.models.model_factory_32 import build_model as build_model32
from utils_svces.models.model_factory_224 import build_model as build_model224
from torchvision import models
import torch.nn as nn
from torchvision import models as torch_models

from utils_svces.model_normalization import Cifar10Wrapper, Cifar100Wrapper, SVHNWrapper,\
    ImageNetWrapper, RestrictedImageNetWrapper, BigTransferWrapper
from utils_svces.temperature_wrapper import TemperatureWrapper
import utils_svces.models.ebm_wrn as wrn
from utils_svces.models.big_transfer_factory import build_model_big_transfer
from utils_svces.datasets.paths import get_CIFAR10_path, get_imagenet_path


def load_non_native_model(type, folder, device):
    if 'Madry' in type:
        from robustness import model_utils, datasets

        class MadryWrapper(torch.nn.Module):
            def __init__(self, model, normalizer):
                super().__init__()
                self.model = model
                self.normalizer = normalizer

            def forward(self, img):
                normalized_inp = self.normalizer(img)
                output = self.model(normalized_inp, with_latent=False,
                                    fake_relu=False, no_relu=False)
                return output

        if type == 'MadryRestrictedImageNet50':
            dataset = datasets.DATASETS['restricted_imagenet'](get_imagenet_path())
            resume_path = f'RestrictedImageNetModels/MadryModels/ResNet50/{folder}.pt'
        elif type == 'MadryImageNet50':
            dataset = datasets.DATASETS['imagenet'](get_imagenet_path())
            resume_path = f'ImageNetModels/MadryModels/ResNet50/{folder}.pt'
        elif type == 'MadryCifar50':
            dataset = datasets.DATASETS['cifar'](get_CIFAR10_path())
            resume_path = f'Cifar10Models/MadryModels/ResNet50/{folder}.pt'
        else:
            raise NotImplementedError()

        model_kwargs = {
            'arch': 'resnet50',
            'dataset': dataset,
            'resume_path': resume_path,
            'parallel' : False
        }
        model_madry, _ = model_utils.make_and_restore_model(**model_kwargs)

        model = MadryWrapper(model_madry.model, model_madry.normalizer)
        model.to(device)
        model.eval()
    elif type == 'PytorchResNet50':
        model = torch_models.resnet50()
        state_dict_file = f'ImageNetModels/PytorchModels/ResNet50/{folder}.pt'
        state_dict = torch.load(state_dict_file, map_location=device)
        model.load_state_dict(state_dict)
        model = ImageNetWrapper(model)
        model = model.to(device)
        model.eval()
    elif type == 'TRADESReference':
        model = ResNet50(num_classes=10)
        state_dict_file = f'{folder}.pt'
        state_dict = torch.load(state_dict_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    elif type == 'Carmon':
        model = WideResNetCarmon(num_classes=10, depth=28, widen_factor=10)
        state_dict_file = f'Cifar10Models/{folder}.pt'
        checkpoint = torch.load(state_dict_file, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint )
        num_classes = checkpoint.get('num_classes', 10)
        normalize_input = checkpoint.get('normalize_input', False)
        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    else:
        raise ValueError('Type not supported')

    return model


def get_filename(folder, architecture_folder, checkpoint, load_temp):
    if load_temp:
        load_folder_name = f'_temp_{folder}'
    else:
        load_folder_name = f'{folder}'

    if not  checkpoint.isnumeric():
        state_dict_file = f'{architecture_folder}/{load_folder_name}/{checkpoint}.pth'
    else:
        epoch = int(checkpoint)
        state_dict_file = f'{architecture_folder}/{load_folder_name}/checkpoints/{epoch}.pth'
    return state_dict_file


non_native_model = ['PytorchResNet50', 'Madry50', 'TRADESReference', 'MadryRestrictedImageNet50', 'MadryImageNet50', 'Carmon']

def load_cifar_family_model(type, folder, checkpoint, device, dataset_dir, num_classes, load_temp=False, model_params=None):
    model, model_folder_post, _, img_size = build_model32(type, num_classes, model_params=model_params)
    state_dict_file = get_filename(folder, os.path.join(dataset_dir, model_folder_post), checkpoint, load_temp)
    state_dict = torch.load(state_dict_file, map_location=device)
    model.load_state_dict(state_dict)
    return model

def load_big_transfer_model(type, folder, checkpoint, device, dataset_dir, num_classes, load_temp=False, model_params=None):
    model, model_folder_post = build_model_big_transfer(type, num_classes)
    state_dict_file = get_filename(folder, os.path.join(dataset_dir, model_folder_post), checkpoint, load_temp)
    state_dict = torch.load(state_dict_file, map_location=device)
    model.load_state_dict(state_dict)
    return model

def load_imagenet_family_model(type, folder, checkpoint, device, dataset_dir, num_classes, load_temp=False, model_params=None):
    model, model_folder_post, _ = build_model224(type, num_classes, **model_params)
    state_dict_file = get_filename(folder, f'{dataset_dir}/{model_folder_post}', checkpoint, load_temp)
    state_dict = torch.load(state_dict_file, map_location=device)
    model.load_state_dict(state_dict)

    return model

def load_model(type, folder, checkpoint, temperature, device, dataset='cifar10', load_temp=False,  model_params=None):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        dataset_dir = 'Cifar10Models'
        num_classes = 10
        model_family = 'Cifar32'
    elif dataset == 'cifar100':
        dataset_dir = 'Cifar100Models'
        num_classes = 100
        model_family = 'Cifar32'
    elif dataset == 'svhn':
        dataset_dir = 'SVHNModels'
        num_classes = 10
        model_family = 'Cifar32'
    elif dataset == 'tinyImageNet':
        dataset_dir = 'TinyImageNetModels'
        num_classes = 200
        model_family = 'ImageNet224'
    elif dataset == 'restrictedimagenet':
        #dataset_dir = 'RestrictedImageNetModels'
        dataset_dir = 'RestrictedImageNetModels'
        num_classes = 9
        model_family = 'ImageNet224'
    elif dataset == 'imagenet':
        dataset_dir = 'ImageNetModels'
        num_classes = 1000
        model_family = 'ImageNet224'
    elif dataset == 'imagenet100':
        dataset_dir = 'ImageNet100Models'
        num_classes = 100
        model_family = 'ImageNet224'
    elif dataset == 'pets':
        dataset_dir = 'PetsModels'
        num_classes = 37
        model_family = 'ImageNet224'
    elif dataset == 'flowers':
        dataset_dir = 'FlowersModels'
        num_classes = 102
        model_family = 'ImageNet224'
    elif dataset == 'cars':
        dataset_dir = 'CarsModels'
        num_classes = 196
        model_family = 'ImageNet224'
    elif dataset == 'food-101':
        dataset_dir = 'Food-101Models'
        num_classes = 101
        model_family = 'ImageNet224'
    elif dataset == 'lsun_scenes':
        dataset_dir = 'LSUNScenesModels'
        num_classes = 10
        model_family = 'ImageNet224'
    else:
        raise ValueError('Dataset not supported')

    if type in non_native_model:
        model = load_non_native_model(type, folder, device)
        if temperature is not None:
            model = TemperatureWrapper(model, temperature)
        return model

    if 'BiT' in type:
        model = load_big_transfer_model(type, folder, checkpoint, device, dataset_dir, num_classes, load_temp=load_temp)
        model = BigTransferWrapper(model)
    else:
        if model_family == 'Cifar32':
            model = load_cifar_family_model(type, folder, checkpoint, device, dataset_dir, num_classes,
                                            load_temp=load_temp, model_params=model_params)
        elif model_family == 'ImageNet224':
            model = load_imagenet_family_model(type, folder, checkpoint, device, dataset_dir, num_classes,
                                               load_temp=load_temp, model_params=model_params)
        else:
            raise ValueError()

        if dataset == 'cifar10':
            model = Cifar10Wrapper(model)
        elif dataset == 'cifar100':
            model = Cifar100Wrapper(model)
        elif dataset == 'svhn':
            model = SVHNWrapper(model)
        elif dataset == 'tinyimagenet':
            model = Cifar100Wrapper(model)
        elif dataset == 'imagenet':
            model = ImageNetWrapper(model)
        elif dataset == 'restrictedimagenet':
            model = RestrictedImageNetWrapper(model)
        elif dataset == 'imagenet100':
            model = ImageNetWrapper(model)
        elif dataset == 'pets':
            model = ImageNetWrapper(model)
        elif dataset == 'food-101':
            model = ImageNetWrapper(model)
        elif dataset == 'cars':
            model = ImageNetWrapper(model)
        elif dataset == 'flowers':
            model = ImageNetWrapper(model)
        elif dataset == 'lsun_scenes':
            model = ImageNetWrapper(model)
        else:
            raise ValueError('Dataset not supported')

    model.to(device)

    if temperature is not None:
        model = TemperatureWrapper(model, temperature)

    model.eval()
    return model