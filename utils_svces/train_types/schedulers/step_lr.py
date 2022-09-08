""" Step Scheduler

Basic step LR schedule with warmup, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch

from .scheduler import Scheduler


class StepLRScheduler(Scheduler):
    """
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_epochs: list,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            decay_factor = 1.
            for epoch in self.decay_epochs:
                if t >= epoch:
                    decay_factor *= self.decay_rate
            lrs = [v * decay_factor for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: float):
        return self._get_lr(epoch)
