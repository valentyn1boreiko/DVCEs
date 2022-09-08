import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import logits_diff_loss, conf_diff_loss, confidence_loss, reduce, log_confidence_loss


class AdversarialAttack():
    def __init__(self, loss, num_classes, model=None, save_trajectory=False):
        #loss should either be a string specifying one of the predefined loss functions
        #OR
        #a custom loss function taking 4 arguments as train_loss class
        self.loss = loss
        self.save_trajectory = save_trajectory
        self.last_trajectory = None
        self.num_classes = num_classes
        if model is not None:
            self.model = model
        else:
            self.model = None

    def __call__(self, *args, **kwargs):
        return self.perturb(*args,**kwargs)

    def set_loss(self, loss):
        self.loss = loss

    def set_model(self, model):
        self.model = model

    def _get_loss_f(self, x, y, targeted, reduction):
        #x, y original ref_data / target
        #targeted whether to use a targeted attack or not
        #reduction: reduction to use: 'sum', 'mean', 'none'
        if isinstance(self.loss, str):
            if self.loss.lower() in ['crossentropy', 'ce']:
                if not targeted:
                    l_f = lambda data, data_out: -F.cross_entropy(data_out, y, reduction=reduction)
                else:
                    l_f = lambda data, data_out: F.cross_entropy(data_out, y, reduction=reduction )
            elif self.loss.lower() =='kl':
                if not targeted:
                    l_f = lambda data, data_out: -reduce(F.kl_div(torch.log_softmax(data_out,dim=1), y, reduction='none').sum(dim=1), reduction)
                else:
                    l_f = lambda data, data_out: reduce(F.kl_div(torch.log_softmax(data_out,dim=1), y, reduction='none').sum(dim=1), reduction)
            elif self.loss.lower() == 'logitsdiff':
                if not targeted:
                    y_oh = F.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: -logits_diff_loss(data_out, y_oh, reduction=reduction)
                else:
                    y_oh = F.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: logits_diff_loss(data_out, y_oh, reduction=reduction)
            elif self.loss.lower() == 'conf':
                if not targeted:
                    l_f = lambda data, data_out: confidence_loss(data_out, y, reduction=reduction)
                else:
                    l_f = lambda data, data_out: -confidence_loss(data_out, y, reduction=reduction)
            elif self.loss.lower() == 'log_conf':
                if not targeted:
                    l_f = lambda data, data_out: log_confidence_loss(data_out, y, reduction=reduction)
                else:
                    l_f = lambda data, data_out: -log_confidence_loss(data_out, y, reduction=reduction)
            elif self.loss.lower() == 'confdiff':
                if not targeted:
                    y_oh = F.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: -conf_diff_loss(data_out, y_oh, reduction=reduction)
                else:
                    y_oh = F.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: conf_diff_loss(data_out, y_oh, reduction=reduction)
            else:
                raise ValueError(f'Loss {self.loss} not supported')
        else:
            #custom 5 argument loss
            #(x_adv, x_adv_out, x, y, reduction)
            l_f = lambda data, data_out: self.loss(data, data_out, x, y, reduction=reduction)

        return l_f

    def get_config_dict(self):
        raise NotImplementedError()

    def get_last_trajectory(self):
        #output dimension: (iterations, batch_size, img_dimension)
        if not self.save_trajectory or self.last_trajectory is None:
            raise AssertionError()
        else:
            return self.last_trajectory

    def _get_trajectory_depth(self):
        raise NotImplementedError()

    def _check_model(self):
        if self.model is None:
            raise RuntimeError('Attack density_model not set')

    def perturb(self, x, y, targeted=False, x_init=None):
        #force child class implementation
        raise NotImplementedError()