import socket
print()
import os


def get_base_data_dir():
    path = '/scratch/datasets/'
    return path

def get_svhn_path():
    return os.path.join(get_base_data_dir(),  'SVHN')

def get_fundusKaggle_path(background_subtraction=False):
    return ''

def get_CIFAR10_path():
    return os.path.join(get_base_data_dir(),  'CIFAR10')

def get_CIFAR100_path():
    return os.path.join(get_base_data_dir(),  'CIFAR100')

def get_CIFAR10_C_path():
    return os.path.join(get_base_data_dir(),  'CIFAR-10-C')

def get_CIFAR100_C_path():
    return os.path.join(get_base_data_dir(),  'CIFAR-100-C')

def get_CINIC10_path():
    return os.path.join(get_base_data_dir(),  'cinic_10')


def get_celebA_path():
    return get_base_data_dir()

def get_stanford_cars_path():
    return os.path.join(get_base_data_dir(),  'stanford_cars')

def get_flowers_path():
    return os.path.join(get_base_data_dir(),  'flowers')

def get_pets_path():
    return os.path.join(get_base_data_dir(),  'pets')

def get_food_101N_path():
    return os.path.join(get_base_data_dir(),   'Food-101N', 'Food-101N_release')

def get_food_101_path():
    return os.path.join(get_base_data_dir(),   'Food-101')

def get_fgvc_aircraft_path():
    return os.path.join(get_base_data_dir(),  'FGVC/fgvc-aircraft-2013b')

def get_cub_path():
    return os.path.join(get_base_data_dir(),  'CUB')

def get_LSUN_scenes_path():
    return os.path.join(get_base_data_dir(),  'LSUN_scenes')


def get_tiny_images_files(shuffled=True):
    if shuffled == True:
        raise NotImplementedError()
    else:
        return get_base_data_dir() + '80M Tiny Images/tiny_images.bin'

def get_tiny_images_lmdb():
    raise NotImplementedError()

def get_imagenet_path(data_folder=None):
    path = os.path.join(get_base_data_dir(), 'imagenet')
    return path

def get_imagenet_o_path():
    return os.path.join(get_base_data_dir(), 'imagenet-o')

def get_openimages_path():
    path = '/home/scratch/datasets/openimages/'
    return path


def get_tiny_imagenet_path():
    return get_base_data_dir() + 'TinyImageNet/tiny-imagenet-200/'