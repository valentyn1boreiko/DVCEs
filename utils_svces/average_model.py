import torch
import math
from torch.nn import Module
from copy import deepcopy
from torch.nn.modules.batchnorm import _BatchNorm

class AveragedModel(Module):
    """
    Modified AveragedModel from torch swa_utils that supports EMA and SWA updates and batch norm averaging
    """
    def __init__(self, model, avg_type='ema', ema_decay=0.990, avg_batchnorm=False, device=None):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))

        assert avg_type in ['ema', 'swa']
        self.avg_type = avg_type
        self.ema_decay = ema_decay
        self.avg_batchnorm = avg_batchnorm

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        n = self.n_averaged.item()
        if self.avg_type == 'ema':
            decay = min(
                self.ema_decay,
                (1 + n) / (10 + n)
            )
            avg_fn = lambda averaged_model_parameter, model_parameter: \
                decay * averaged_model_parameter + (1.0 - decay) * model_parameter
        elif self.avg_type == 'swa':
            avg_fn = lambda averaged_model_parameter, model_parameter: \
                (model_parameter - averaged_model_parameter) / (n + 1)
        else:
            raise NotImplementedError()

        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if n == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(avg_fn(p_swa.detach(), p_model_,))

        if self.avg_batchnorm:
            for avg_mod, model_mod in zip(self.module.modules(), model.modules()):
                if issubclass(type(model_mod), _BatchNorm):
                    device = avg_mod.running_mean.device
                    mean_model_ = model_mod.running_mean.detach().to(device)
                    var_model_ = model_mod.running_var.detach().to(device)
                    if n == 0:
                        avg_mod.running_mean.detach().copy_(mean_model_)
                        avg_mod.running_var.detach().copy_(var_model_)
                    else:
                        avg_mod.running_mean.detach().copy_(
                            avg_fn(avg_mod.running_mean.detach(), mean_model_))
                        avg_mod.running_var.detach().copy_(
                            avg_fn(avg_mod.running_var.detach(), var_model_))

        self.n_averaged += 1