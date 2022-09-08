import torch.nn.functional as F
from .fmix_utils import sample_mask
import torch
from .mixed_sample_data_augmentation import MixedSampleDataAugmentation
from ..train_loss import MinMaxLoss, TrainLoss

class FMixLoss(MinMaxLoss):
    def __init__(self, base_loss, lam=None, index=None, reformulate=False, log_stats=False, name_prefix=None):
        name = 'FMix_' + base_loss.name
        super().__init__(name, expected_format='logits', log_stats=log_stats, name_prefix=name_prefix)
        self.index = index
        self.lam = lam
        self.reformulate = reformulate
        self.base_loss = base_loss

    def inner_max(self, data, target):
        return data

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        assert self.index is not None
        assert self.lam is not None

        if not self.reformulate:
            y2 = y[self.index]
            loss_expanded = self.base_loss(data, model_out, orig_data, y, reduction='none') * self.lam\
                            + self.base_loss(data, model_out, orig_data, y2, reduction='none') * (1 - self.lam)
        else:
            loss_expanded = self.base_loss(data, model_out, orig_data, y, reduction='none')

        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class FMix(MixedSampleDataAugmentation):
    r""" FMix augmentation
        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims. -1 computes on the fly
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].

    """
    def __init__(self, base_loss, decay_power=3, alpha=1, size=(-1, -1), max_soft=0.0, reformulate=False,
                 log_stats=False, name_prefix=None):
        self.decay_power = decay_power
        self.reformulate = reformulate
        self.size = size
        self.alpha = alpha
        self.max_soft = max_soft

        loss = FMixLoss(base_loss, reformulate=reformulate, log_stats=log_stats, name_prefix=name_prefix)
        super().__init__(loss)

    def apply_mix(self, x):
        size = []
        for i, s in enumerate(self.size):
            if s != -1:
                size.append(s)
            else:
                size.append(x.shape[i+2])

        lam, mask = sample_mask(self.alpha, self.decay_power, size, self.max_soft, self.reformulate)
        index = torch.randperm(x.size(0)).to(x.device)
        mask = torch.from_numpy(mask).float().to(x.device)

        if len(self.size) == 1 and x.ndim == 3:
            mask = mask.unsqueeze(2)

        # Mix the images
        x_mix = mask * x + (1 - mask) * x[index]

        self.loss.od_weight = lam
        self.loss.index = index

        return x_mix


