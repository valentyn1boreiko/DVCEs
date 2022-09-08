from torchvision import transforms
import torch
from utils_svces.datasets.autoaugment import ImageNetPolicy

# lighting transform
# https://git.io/fhBOc
IMAGENET_PCA = {
    'eigval':torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


ImageNet_mean_int = ( int( 255 * 0.485), int(255 * 0.456), int(255 * 0.406))


def get_imageNet_augmentation(type='default', out_size=224, config_dict=None):
    if type == 'none' or type is None:
        transform_list = [
            transforms.Resize((out_size,out_size)),
            transforms.ToTensor()
        ]
        transform = transforms.Compose(transform_list)
        return transform
    elif type == 'madry':
        # Special transforms for ImageNet(s)
        """
        Standard training ref_data augmentation for ImageNet-scale datasets: Random crop,
        Random flip, Color Jitter, and Lighting Transform (see https://git.io/fhBOc)
        """
        transform_list = [
            transforms.RandomResizedCrop(out_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),]

        transform_list.append(transforms.ToTensor())
        transform_list.append(Lighting(0.05, IMAGENET_PCA['eigval'],
                     IMAGENET_PCA['eigvec']))
        transform = transforms.Compose(transform_list)
        return transform
    elif type == 'test' or type is None:
        transform_list = [
            transforms.Resize(int(256/224 * out_size)),
            transforms.CenterCrop(out_size),
        ]
    elif type == 'default':
        transform_list = [
            transforms.transforms.RandomResizedCrop(out_size),
            transforms.RandomHorizontalFlip(),
        ]
    elif type == 'big_transfer' or type == 'big_transfer_128':
        if type == 'big_transfer':
            if out_size != 480:
                print(f'Out size of {out_size} detected but Big Transfer is supposed to be used with 480')
                pre_crop_size = int(out_size * (512/480))
            else:
                pre_crop_size = 512
        else:
            if out_size != 128:
                print(f'Out size of {out_size} detected but Big Transfer 128 is supposed to be used with 128')
                pre_crop_size = int(out_size * (160 / 128))
            else:
                pre_crop_size = 160

        print(f'BigTransfer augmentation: Pre crop {pre_crop_size} - Out Size {out_size}')
        transform_list = [
            transforms.transforms.Resize((pre_crop_size, pre_crop_size)),
            transforms.transforms.RandomCrop((out_size, out_size)),
            transforms.transforms.RandomHorizontalFlip(),
        ]
    elif type == 'autoaugment':
        transform_list = [
            transforms.RandomResizedCrop(out_size),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(fillcolor=ImageNet_mean_int),
        ]
    else:
        raise ValueError(f'augmentation type - {type} - not supported')

    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    if config_dict is not None:
        config_dict['type'] = type
        config_dict['Output size'] = out_size

    return transform