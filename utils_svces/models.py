import torch
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from sklearn.cluster import KMeans

import numpy as np
from sklearn.decomposition import PCA


class Metric(nn.Module):
    '''
        Abstract class that defines the concept of a metric. It is needed
        to define mixture models with different metrics.
        In the paper we use the PCAMetric
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, dim=None):
        pass
    
    def __add__(self, other):
        return SumMetric(self, other)
    
    def __rmul__(self, scalar):
        return ScaleMetric(scalar, self)
    
    
class SumMetric(Metric):
    def __init__(self, metric1, metric2):
        super().__init__()
        self.metric1 = metric1
        self.metric2 = metric2
        
    def forward(self, x, y, dim=None):
        return self.metric1(x, y, dim=dim) + self.metric2(x, y, dim=dim)
    
    
class ScaleMetric(Metric):
    def __init__(self, metric1, factor):
        super().__init__()
        self.metric1 = metric1
        self.factor = factor
        
    def forward(self, x, y, dim=None):
        return self.factor * self.metric1(x, y, dim=dim)


class LpMetric(Metric):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
        self.norm_const = 0.
        
    def forward(self, x, y, dim=None):
        return (x-y).norm(p=self.p, dim=dim)
    

class PerceptualMetric(Metric):
    def __init__(self, model, p=2, latent_dim=122880, indices=None):
        super().__init__()
        self.model = model
        self.p = p
        self.norm_const = 0.
        
        self.latent_dim = latent_dim
        reduced_latent_dim = int(0.01*latent_dim)
        
        if indices is None:
            self.indices = sorted(np.random.choice(latent_dim, size=reduced_latent_dim, replace=False))
        else:
            self.indices = indices
        
    def forward(self, x, y, dim=None):
        return (self.model(x)[:,self.indices][None,:,:]
                -self.model(y)[:,self.indices][:,None,:]).norm(p=self.p, dim=dim)

    
class PerceptualPCA(Metric):
    def __init__(self, model, pca, indices=None):
        super().__init__()
        self.model = model
        
        self.pca = pca
        
        if indices is None:
            self.indices = sorted(np.random.choice(latent_dim, size=reduced_latent_dim, replace=False))
        else:
            self.indices = indices
        
        
    def forward(self, x, y, dim=None):
        return self.pca(self.model(x)[:,self.indices][None,:,:],
                        self.model(y)[:,self.indices][:,None,:], dim=dim)

    
class PCAMetric(Metric):
    def __init__(self, X, p=2, min_sv_factor=100., covar=None):
        super().__init__()
        self.p = p
        
        if covar is None:
            X = np.array(X)
            pca = PCA()
            pca.fit(X)

            self.comp_vecs = nn.Parameter(torch.tensor(pca.components_), requires_grad=False)
            self.singular_values = torch.tensor(pca.singular_values_)
        else:
            singular_values, comp_vecs = np.linalg.eig(covar)

            self.comp_vecs = nn.Parameter(torch.tensor(comp_vecs, dtype=torch.float), requires_grad=False)
            self.singular_values = torch.tensor(singular_values, dtype=torch.float)
            
        self.min_sv = self.singular_values[0] / min_sv_factor
        self.singular_values[self.singular_values<self.min_sv] = self.min_sv
        self.singular_values = nn.Parameter(self.singular_values, requires_grad=False)
        self.singular_values_sqrt = nn.Parameter(self.singular_values.sqrt(), requires_grad=False)
        
        self.norm_const = self.singular_values.log().sum()
        
    def forward(self, x, y, dim=None):
        rotated_dist = torch.einsum("ijk,lk->ijl", (x-y, self.comp_vecs))
        rescaled_dist = rotated_dist / self.singular_values_sqrt[None,None,:]
        return rescaled_dist.norm(dim=2, p=self.p)


    
class MyPCA():
    '''
        A helper class that is used for adversarial attacks in a Mahalanobis Metric
    '''
    def __init__(self, comp_vecs, singular_values, shape):
        self.comp_vecs = comp_vecs
        self.comp_vecs_inverse = self.comp_vecs.inverse()
        self.singular_values = singular_values
        self.singular_values_sqrt = singular_values.sqrt()
        self.shape = tuple(shape)
        self.D = torch.tensor(shape).prod().item()
        
    def inv_trans(self, x):
        x = ( (x * self.singular_values_sqrt[None,:] ) @ self.comp_vecs_inverse )
        return x.view(tuple([x.shape[0]]) + self.shape)
    
    def trans(self, x):
        x = x.view(-1, self.D)
        return ( (x@self.comp_vecs) / self.singular_values_sqrt[None,:] )

    
class LeNet(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

        self.normalize = normalize
        self.mean = 0.1307
        self.std = 0.3081

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
