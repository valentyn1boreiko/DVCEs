# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn as nn


class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalActNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data.zero_()
        self.init = False

    def forward(self, x, y):
        if self.init:
            scale, bias = self.embed(y).chunk(2, dim=-1)
            return x * scale[:, :, None, None] + bias[:, :, None, None]
        else:
            m, v = torch.mean(x, dim=(0, 2, 3)), torch.var(x, dim=(0, 2, 3))
            std = torch.sqrt(v + 1e-5)
            scale_init = 1. / std
            bias_init = -1. * m / std
            self.embed.weight.data[:, :self.num_features] = scale_init[None].repeat(self.num_classes, 1)
            self.embed.weight.data[:, self.num_features:] = bias_init[None].repeat(self.num_classes, 1)
            self.init = True
            return self(x, y)


logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class ContinuousConditionalActNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        del num_classes
        self.num_features = num_features
        self.embed = nn.Sequential(nn.Linear(1, 256),
                                   nn.ELU(inplace=True),
                                   nn.Linear(256, 256),
                                   nn.ELU(inplace=True),
                                   nn.Linear(256, self.num_features*2),
                                   )
        
    def forward(self, x, y):
        scale, bias = self.embed(y.unsqueeze(-1)).chunk(2, dim=-1)
        return x * scale[:, :, None, None] + bias[:, :, None, None]



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm=None, leak=.2):
        super(wide_basic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)

        return out


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)
    elif norm == "act":
        return ActNorm(n_filters, False)


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, input_channels=3,
                 sum_pool=False, norm=None, leak=.2, dropout_rate=0.0):
        super(Wide_ResNet, self).__init__()
        self.leak = leak
        self.in_planes = 16
        self.sum_pool = sum_pool
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(input_channels, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = get_norm(nStages[3], self.norm)
        self.last_dim = nStages[3]
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, norm=self.norm))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, vx=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        if self.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)
        else:
            out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out

    
class EBM_wrapper(nn.Module):
    def __init__(self, model):
        super(EBM_wrapper, self).__init__()
        self.sigma = 3e-2
        self.model = model
        
    def forward(self, x):
        x = 2*x-1 + self.sigma*torch.randn_like(x)
        return self.model.classify(x)
    
    def energy(self, x, y=None):
        x = 2*x-1 + self.sigma*torch.randn_like(x)
        penult_z = self.model.f(x)
        return self.model.energy_output(penult_z).squeeze()
    
    
class EBM(nn.Module):
    def __init__(self, depth=28, width=2, norm=None):
        super(EBM, self).__init__()
        self.f = Wide_ResNet(depth, width, norm=norm)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, 10)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z)


class CCF(EBM):
    def __init__(self, depth=28, width=2, norm=None):
        super(CCF, self).__init__(depth, width, norm=norm)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return torch.gather(logits, 1, y[:, None])
    
    
class gradient_attack_wrapper(nn.Module):
    def __init__(self, model):
        super(gradient_attack_wrapper, self).__init__()
        self.model = model.eval()

    def forward(self, x):
        x = x - 0.5
        x = x / 0.5
        x.requires_grad_()
        out = self.model.refined_logits(x)
        return out

    def eval(self):
        return self.model.eval()
        
        
class DummyModel(nn.Module):
    def __init__(self, f, n_steps_refine=0):
        super(DummyModel, self).__init__()
        self.f = f
        self.sgld_lr = 1.
        self.sgld_std = 1e-2
        self.n_dup_chains = 5
        self.n_steps_refine = n_steps_refine
        self.sigma = 3e-2
        self.detach=True
        
    def logits(self, x):
        return self.f.classify(x)

    def refined_logits(self, x, n_steps=None):
        n_steps = self.n_steps_refine if n_steps is None else n_steps
        xs = x.size()
        dup_x = x.view(xs[0], 1, xs[1], xs[2], xs[3]).repeat(1, self.n_dup_chains, 1, 1, 1)
        dup_x = dup_x.view(xs[0] * self.n_dup_chains, xs[1], xs[2], xs[3])
        dup_x = dup_x + torch.randn_like(dup_x) * self.sigma
        refined = self.refine(dup_x, n_steps=n_steps, detach=self.detach)
        logits = self.logits(refined)
        logits = logits.view(x.size(0), self.n_dup_chains, logits.size(1))
        logits = logits.mean(1)
        return logits

    def classify(self, x):
        logits = self.logits(x)
        pred = logits.max(1)[1]
        return pred

    def logpx_score(self, x):
        # unnormalized logprob, unconditional on class
        return self.f(x)

    def refine(self, x, n_steps=None, detach=True):
        n_steps = self.n_steps_refine if n_steps is None else n_steps
        # runs a markov chain seeded at x, use n_steps=10
        with torch.enable_grad():
            x_k = torch.autograd.Variable(x, requires_grad=True) if detach else x
            # sgld
            for k in range(n_steps):
                f_prime = torch.autograd.grad(self.f(x_k).sum(), [x_k], retain_graph=True)[0]
                x_k.data += f_prime + self.sgld_std * torch.randn_like(x_k)
        final_samples = x_k.detach() if detach else x_k
        return final_samples

    def grad_norm(self, x):
        x_k = torch.autograd.Variable(x, requires_grad=True)
        with torch.enable_grad():
            f_prime = torch.autograd.grad(self.f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    def logpx_delta_score(self, x, n_steps=None):
        n_steps = self.n_steps_refine if n_steps is None else n_steps
        # difference in logprobs from input x and samples from a markov chain seeded at x
        #
        init_scores = self.f(x)
        x_r = self.refine(x, n_steps=n_steps)
        final_scores = self.f(x_r)
        # for real ref_data final_score is only slightly higher than init_score
        return init_scores - final_scores

    def logp_grad_score(self, x):
        return -self.grad_norm(x)
    
