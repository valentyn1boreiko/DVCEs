import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://github.com/tml-epfl/adv-training-corruptions/blob/ef9aed5322b33020912163558e5bcd76b22ebdb8/models.py

class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, n_cls, model_width=64, cuda=True, cifar_norm=True):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = model_width
        self.avg_preact = None
        if cifar_norm:
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)

        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, model_width, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, model_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * model_width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * model_width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * model_width, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(8 * model_width * block.expansion)
        self.linear = nn.Linear(8 * model_width * block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, delta=None, ri=-1):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        if delta is None:
            out = self.normalize(x)
            out = self.conv1(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif ri == -1:
            out = self.normalize(torch.clamp(x + delta[0], 0, 1))
            out = self.conv1(out) + delta[1]
            out = self.layer1(out) + delta[2]
            out = self.layer2(out) + delta[3]
            out = self.layer3(out) + delta[4]
            out = self.layer4(out) + delta[5]
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)  # + delta[6]
        else:
            out = self.normalize(torch.clamp(x + delta[0], 0, 1)) if ri == 0 else self.normalize(torch.clamp(x, 0, 1))
            out = self.conv1(out + delta[1]) if ri == 1 else self.conv1(out)
            out = self.layer1(out + delta[2]) if ri == 2 else self.layer1(out)
            out = self.layer2(out + delta[3]) if ri == 3 else self.layer2(out)
            out = self.layer3(out + delta[4]) if ri == 4 else self.layer3(out)
            out = self.layer4(out + delta[5]) if ri == 5 else self.layer4(out)
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out + delta[6]) if ri == 6 else self.linear(out)

        return out


def PreActResNet18(n_cls, model_width=64, cuda=True, cifar_norm=True):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, model_width=model_width, cuda=cuda,
                        cifar_norm=cifar_norm)

class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std