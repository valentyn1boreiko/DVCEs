import numpy as np
import torch
from .mixed_sample_data_augmentation import MixedSampleDataAugmentation
from ..train_loss import MinMaxLoss, TrainLoss

class MixupLoss(MinMaxLoss):
    def __init__(self, base_loss, lam=None, index=None, log_stats=False, name_prefix=None):
        name = 'Mixup_' + base_loss.name
        super().__init__(name, expected_format='logits', log_stats=log_stats, name_prefix=name_prefix)
        self.index = index
        self.lam = lam
        self.base_loss = base_loss

    def inner_max(self, data, target):
        return data

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        assert self.index is not None
        assert self.lam is not None

        y2 = y[self.index]
        loss_expanded = self.lam * self.base_loss(data, model_out, orig_data, y, reduction='none')\
                        + (1. - self.lam) * self.base_loss(data, model_out, orig_data, y2, reduction='none')

        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class Mixup(MixedSampleDataAugmentation):
    def __init__(self, base_loss, alpha=1, log_stats=False, name_prefix=None):
        self.alpha = alpha
        loss = MixupLoss(base_loss, log_stats=log_stats, name_prefix=name_prefix)
        super().__init__(loss)

    def apply_mix(self, x):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        index = torch.randperm(x.size(0)).to(x.device)
        x_mix = lam * x + (1 - lam) * x[index, :]

        self.loss.od_weight = lam
        self.loss.index = index

        return x_mix


