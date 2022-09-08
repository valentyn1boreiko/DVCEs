import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tp_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return .5 * (x + delta) * (1 - ind1) * (1 - ind2) + x * ind2

def tp_smoothed_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return (x + delta) ** 2 / (4 * delta) * (1 - ind1) * (1 - ind2) + x * ind2


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std


class ReluWithStats(nn.Module):
    def __init__(self):
        super(ReluWithStats, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []

    def forward(self, preact):
        if self.collect_preact:
            self.avg_preacts.append(preact.abs().mean().item())
        act = F.relu(preact)
        return act


class ModuleWithStats(nn.Module):
    def __init__(self):
        super(ModuleWithStats, self).__init__()

    def forward(self, x):
        for layer in self._model:
            if type(layer) == ReluWithStats:
                layer.avg_preacts = []

        out = self._model(x)

        avg_preacts_all = [layer.avg_preacts for layer in self._model if type(layer) == ReluWithStats]
        self.avg_preact = np.mean(avg_preacts_all)
        return out


class Linear(ModuleWithStats):
    def __init__(self, n_cls, shape_in):
        super(Linear, self).__init__()
        d = int(np.prod(shape_in[1:]))
        self._model = nn.Sequential(
            Flatten(),
            nn.Linear(d, n_cls)
        )


class FC(ModuleWithStats):
    def __init__(self, n_cls, shape_in, n_hl, n_hidden):
        super(FC, self).__init__()
        fc_layers = []
        for i_layer in range(n_hl):
            n_in = np.prod(shape_in[1:]) if i_layer == 0 else n_hidden
            n_out = n_hidden
            fc_layers += [nn.Linear(n_in, n_out), ReluWithStats()]
        self._model = nn.Sequential(
            Flatten(),
            *fc_layers,
            ReluWithStats(),
            nn.Linear(n_hidden, n_cls)
        )


class CNNBase(ModuleWithStats):
    def __init__(self):
        super(CNNBase, self).__init__()


class CNN(CNNBase):
    def __init__(self, n_cls, shape_in, n_conv, n_filters):
        super(CNN, self).__init__()
        input_size = shape_in[2]
        conv_blocks = []
        for i_layer in range(n_conv):
            n_in = shape_in[1] if i_layer == 0 else n_filters
            n_out = n_filters
            conv_blocks += [nn.Conv2d(n_in, n_out, 3, stride=1, padding=1), ReluWithStats()]
        # h_after_conv, w_after_conv = input_size/n_conv, input_size/n_conv
        h_after_conv, w_after_conv = input_size, input_size
        self._model = nn.Sequential(
            *conv_blocks,
            Flatten(),
            nn.Linear(n_filters*h_after_conv*w_after_conv, n_cls)  # a bit large, but ok (163840 parameters for 16 filters)
        )


class CNNLeNet(CNNBase):
    def __init__(self, n_cls, shape_in):
        super(CNNLeNet, self).__init__()
        self._model = nn.Sequential(
            nn.Conv2d(shape_in[1], 16, 4, stride=2, padding=1),
            ReluWithStats(),
            # nn.Dropout2d(p=0.5),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            ReluWithStats(),
            Flatten(),
            # nn.Dropout(p=0.5),
            nn.Linear(32*7*7, 100),
            ReluWithStats(),
            nn.Linear(100, n_cls)
        )


class CNNLeNetGAP(CNNBase):
    def __init__(self, n_cls, shape_in):
        super(CNNLeNetGAP, self).__init__()
        self._model = nn.Sequential(
            nn.Conv2d(shape_in[1], 16, 4, stride=2, padding=1),
            ReluWithStats(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            ReluWithStats(),
            # Flatten(),
            nn.AvgPool2d((7, 7)),  # global average pooling
            Flatten(),
            nn.Linear(32, 100),
            ReluWithStats(),
            nn.Linear(100, n_cls)
        )


class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu'):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            act = F.relu(preact)
        elif self.activation[:6] == '3prelu':
            act = tp_relu(preact, delta=float(self.activation.split('relu')[1]))
        elif self.activation[:8] == '3psmooth':
            act = tp_smoothed_relu(preact, delta=float(self.activation.split('smooth')[1]))
        else:
            assert self.activation[:8] == 'softplus'
            beta = int(self.activation.split('softplus')[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(self.act_function(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, cuda=True, half_prec=False,
        activation='relu', fts_before_bn=False, normal='none'):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = 64
        self.avg_preact = None
        self.activation = activation
        self.fts_before_bn = fts_before_bn
        if normal == 'cifar10':
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
            self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
            print('no input normalization')
        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride, self.activation))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if return_features and self.fts_before_bn:
            return out.view(out.size(0), -1)
        out = F.relu(self.bn(out))
        if return_features:
            return out.view(out.size(0), -1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def PreActResNet18(n_cls, cuda=True, half_prec=False, activation='relu', fts_before_bn=False,
    normal='none'):
    #print('initializing PA RN-18 with act {}, normal {}'.format())
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, cuda=cuda, half_prec=half_prec,
        activation=activation, fts_before_bn=fts_before_bn, normal=normal)


def get_model(model_name, n_cls, half_prec, shapes_dict, n_filters_cnn, n_hidden_fc, activation):
    if model_name == 'resnet18':
        model = PreActResNet18(n_cls, half_prec=half_prec, activation=activation)
    elif model_name == 'lenet':
        model = CNNLeNet(n_cls, shapes_dict)
    elif model_name == 'cnn':
        model = CNN(n_cls, shapes_dict, 1, n_filters_cnn)
    elif model_name == 'fc':
        model = FC(n_cls, shapes_dict, 1, n_hidden_fc)
    elif model_name == 'linear':
        model = Linear(n_cls, shapes_dict)
    else:
        raise ValueError('wrong model')
    return model




    




