#recommended by
#https://arxiv.org/pdf/1511.08861.pdf

#requires pytorch-mssim
#https://github.com/VainF/pytorch-msssim

from pytorch_msssim import SSIM, MS_SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from utils.dataloaders import Cifar10_Inverse_Covariances


class Distance(nn.Module):
    def __init__(self):
        super().__init__()
        
    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self,  X, Y, *args, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        raise NotImplementedError()

    def compute_distance_matrix(self, X, Y, *args, **kwargs):
        D = X.new_zeros((X.shape[0], Y.shape[0]))
        for y_idx in range(Y.shape[0]):
            y = Y[y_idx, :]
            D[:, y_idx] = self.dist(X, y.expand_as(X))
        return D

    
class SquaredEuclideanDistance(Distance):
    def dist(self,  X, Y, *args, **kwargs):
        N = X.shape[0]
        diff = X.view(N, -1)-Y.view(N,-1)
        return torch.sum(diff ** 2, dim=1)


class LPDistance(Distance):
    def __init__(self, p=2.0):
        super().__init__()
        self.p = float(p)

    def dist(self, X, Y, *args, **kwargs):
        #X, Y batches [N, C, H, W]
        N = X.shape[0]
        return (X.view(N, -1)-Y.view(N,-1)).norm(p=self.p, dim=1)

    def get_config(self):
        return {'Distance' : 'LP', 'p': self.p}

class SSIMDistance(Distance):
    def __init__(self, kernel_w=3, sigma=1.5, channels=1):
        super().__init__()
        self.kernel_w = kernel_w
        self.ssim_d = SSIM(win_size=kernel_w, win_sigma=sigma, data_range=1.0, size_average=False, channel=channels)
        self.config =  {'Distance' : 'SSIM', 'win_size': kernel_w, 'win_sigma': sigma}

    def dist(self, X, Y, *args, **kwargs):
        #X, Y batches [N, C, H, W]
        N = X.shape[0]
        d1 = torch.clamp_min(1 - (self.ssim_d(X, Y)).view(N, -1).mean(dim=1), 0.0)
        return d1

class MSSSIMDistance(Distance):
    def __init__(self, kernel_w=3, sigma=1.5, channels=1, weights=None):
        super().__init__()
        self.kernel_w = kernel_w
        self.sigma = sigma

        #number of weights determines the depth of the pyramid
        #standard are 5, too deep for MNist resolution
        if weights is None:
            self.weights = [0.0516, 0.32949, 0.34622, 0.27261]
        else:
            self.weights = weights
        self.ssim_d = MS_SSIM(win_size=kernel_w, win_sigma=sigma, data_range=1.0,
                              channel=channels, weights=self.weights, size_average=False)

        self.config =  {'Distance' : 'SSIM', 'win_size': kernel_w, 'win_sigma': sigma}

    def dist(self, X, Y, *args, **kwargs):
        #X, Y batches [N, C, H, W]
        N = X.shape[0]
        d1 = torch.clamp_min(1 - (self.ssim_d(X, Y)).view(N, -1).mean(dim=1), 0.0)
        return d1


class ReconstructionLoss(Distance):
    def __init__(self, alpha=0.84, kernel_w=3, sigma=1.5, channels=1):
        super.__init__()
        self.alpha = alpha
        self.kernel_w = kernel_w
        self.ssim_d = SSIM(win_size=kernel_w, win_sigma=sigma, data_range=1.0, size_average=False, channel=channels)
        self.config =  {'Distance' : 'L1 + SSIM', 'alpha': alpha, 'win_size': kernel_w, 'win_sigma': sigma}

    def dist(self, X, Y, *args, **kwargs):
        #X, Y batches [N, C, H, W]
        N = X.shape[0]
        numel = X.shape[1] * X.shape[2] * X.shape[3]
        d1 = torch.clamp_min(1 - (self.ssim_d(X, Y)).view(N, -1).mean(dim=1), 0.0)
        d2 = (self.kernel_w / numel) * torch.sum(torch.abs( X.view(N,-1) - Y.view(N,-1)), dim=1)
        return self.alpha * d1 + (1. - self.alpha) * d2

    def get_config(self):
        return self.config


#multiscale
class MSReconstructionLoss(Distance):
    def __init__(self, alpha=0.84, kernel_w=3, sigma=1.5, channels=1, weights=None):
        super().__init__()
        self.alpha = alpha
        self.kernel_w = kernel_w
        self.sigma = sigma

        #number of weights determines the depth of the pyramid
        #standard are 5, too deep for MNist resolution
        if weights is None:
            self.weights = [0.0516, 0.32949, 0.34622, 0.27261]
        else:
            self.weights = weights

        self.config = {'Distance': 'L1 + MS_SSIM', 'alpha': alpha, 
                       'win_size': kernel_w, 'win_sigma': sigma}


    def dist(self, X, Y, *args, **kwargs):
        #X, Y batches [N, C, H, W]

        N = X.shape[0]
        numel = X.shape[1] * X.shape[2] * X.shape[3]

        weights = torch.FloatTensor(self.weights).to(X.device, dtype=X.dtype)
        d1 = torch.clamp_min(ms_ssim(X, Y, win_size=self.kernel_w, win_sigma=self.sigma, data_range=1.0, size_average=False, weights=weights), 0.0)
        d1 = 1. - d1.view(N, -1).mean(dim=1)

        d2 = (self.kernel_w / numel) * torch.sum(torch.abs(X.view(N,-1) - Y.view(N,-1)))

        return self.alpha * d1 + (1. - self.alpha) * d2

    def get_config(self):
        return self.config


