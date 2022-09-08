import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from collections import OrderedDict
import argparse
import os
import sys

sys.path.insert(0, '../../../')
from timm import create_model
from .ghost_bn_old import GhostBN2D_Old


class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()
        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)


class Affine(nn.Module):
    def __init__(self, width, *args, k=1, **kwargs):
        super(Affine, self).__init__()
        self.bnconv = nn.Conv2d(width,
                                width,
                                k,
                                padding=(k - 1) // 2,
                                groups=width,
                                bias=True)

    def forward(self, x):
        return self.bnconv(x)


class EightBN(nn.Module):

    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2, affine=False, sync_stats=False,
                 **kwargs):
        super(EightBN, self).__init__()

        self.bn0 = GhostBN2D_Old(num_features=num_features, *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine,
                                 sync_stats=sync_stats, **kwargs)
        self.bn1 = GhostBN2D_Old(num_features=num_features, *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine,
                                 sync_stats=sync_stats, **kwargs)
        self.bn2 = GhostBN2D_Old(num_features=num_features, *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine,
                                 sync_stats=sync_stats, **kwargs)
        self.bn3 = GhostBN2D_Old(num_features=num_features, *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine,
                                 sync_stats=sync_stats, **kwargs)
        self.bn4 = GhostBN2D_Old(num_features=num_features, *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine,
                                 sync_stats=sync_stats, **kwargs)
        self.bn5 = GhostBN2D_Old(num_features=num_features, *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine,
                                 sync_stats=sync_stats, **kwargs)
        self.bn6 = GhostBN2D_Old(num_features=num_features, *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine,
                                 sync_stats=sync_stats, **kwargs)
        self.bn7 = GhostBN2D_Old(num_features=num_features, *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine,
                                 sync_stats=sync_stats, **kwargs)

        self.bn_type = 'bn0'
        self.aff = Affine(width=num_features, k=1)

    def forward(self, input):
        if self.bn_type == 'bn0':
            input = self.bn0(input)
        elif self.bn_type == 'bn1':
            input = self.bn1(input)
        elif self.bn_type == 'bn2':
            input = self.bn2(input)
        elif self.bn_type == 'bn3':
            input = self.bn3(input)
        elif self.bn_type == 'bn4':
            input = self.bn4(input)
        elif self.bn_type == 'bn5':
            input = self.bn5(input)
        elif self.bn_type == 'bn6':
            input = self.bn6(input)
        elif self.bn_type == 'bn7':
            input = self.bn7(input)

        input = self.aff(input)
        return input


def load_model_ext(modelname):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    # mu = torch.tensor(mean).view(3,1,1).cuda()
    # std = torch.tensor(std).view(3,1,1).cuda()

    if modelname == 'deit_small_patch16_224_adv':
        import models.DeiT_utils.models
        model = create_model(
            'deit_small_patch16_224_adv',
            pretrained=False,
            num_classes=1000,  # args.num_classes
            drop_rate=0.,  # args.drop
            drop_path_rate=.1,  # args.drop_path
            drop_block_rate=None,  #
            # norm = 'layer',
        )
        #a = torch.load(f'{root}/ViTs_vs_CNNs/ckpt/advdeit_small.pth')  # args.ckpt
        #a = torch.load(model_path)
        #model.load_state_dict(a['model'])
        model.cuda()
        model.eval()
        '''model.module.set_sing(True)
        model.module.set_mixup_fn(False)
        model.module.set_mix(False)'''
        model.set_sing(True)
        model.set_mixup_fn(False)
        model.set_mix(False)
        model = normalize_model(model, mean, std)
        model.cuda()
        model.eval()

    elif modelname == 'resnet50gelu':
        import ViTs_vs_CNNs.autoattack.resnet_gbn_gelu_4096 as res
        # from ViTs_vs_CNNs.autoattack.test_autoattack import EightBN
        # norm_layer = nn.BatchNorm2d
        norm_layer = EightBN
        model = res.__dict__['resnet50'](norm_layer=norm_layer)
        #a = torch.load(f'{root}/ViTs_vs_CNNs/ckpt/advres50_gelu.pth')  # args.ckpt
        #a = torch.load(model_path)
        #model.load_state_dict(a['model'])
        model.cuda()
        model.eval()
        model = normalize_model(model, mean, std)
        model.cuda()
        model.eval()
    else:
        raise ValueError(f'unknown model: {modelname}')
    return model

