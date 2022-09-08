import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim

###################################################
class AdversarialNoiseGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        #generate nosie matching the out_size of x
        raise NotImplementedError()

class UniformNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return (self.max - self.min) * torch.rand_like(x) + self.min

class NormalNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, sigma=1.0, mu=0):
        super().__init__()
        self.sigma = sigma
        self.mu = mu

    def forward(self, x):
        return self.sigma * torch.randn_like(x) + self.mu

class CALNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, rho=1, lambda_scheme='normal'):
        super().__init__()
        self.rho = rho
        self.lambda_scheme = lambda_scheme

    def forward(self, x):
        if self.lambda_scheme == 'normal':
            lambda_targets =  x.new_zeros(x.shape[0])
            reject_idcs = lambda_targets < 1
            #rejection sample from truncated normal
            while sum(reject_idcs > 0):
                lambda_targets[reject_idcs] = math.sqrt(self.rho) * torch.randn(sum(reject_idcs), device=x.device).abs() + 1e-8
                reject_idcs =  lambda_targets > 1
        elif self.lambda_scheme == 'uniform':
            lambda_targets = torch.rand(x.shape[0], device=x.device)

        target_dists_sqr = -torch.log( lambda_targets) * self.rho
        dirs = torch.randn_like(x)
        dirs_lengths = torch.norm( dirs.view( x.shape[0], -1)  , dim=1)
        dirs_normalized = dirs / dirs_lengths.view(x.shape[0], 1, 1, 1)
        perts = target_dists_sqr.sqrt().view(x.shape[0], 1, 1, 1) * dirs_normalized
        return perts

