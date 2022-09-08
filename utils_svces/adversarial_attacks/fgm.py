import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .restartattack import RestartAttack
from .utils import project_perturbation, normalize_perturbation, initialize_perturbation

class FGM(RestartAttack):
    #one step attack with l2 or inf norm constraint
    def __init__(self, eps, num_classes, norm='inf', loss='CrossEntropy', restarts=0, init_noise_generator=None,
                 model=None, save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        self.eps = eps
        self.norm = norm
        self.init_noise_generator = init_noise_generator

    def _get_trajectory_depth(self):
        return 2

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'FGM'
        dict['eps'] = self.eps
        dict['norm'] = self.norm
        if isinstance(self.loss, str):
            dict['loss'] = self.loss
        else:
            dict['loss'] = 'custom'
        dict['restarts'] = self.restarts
        return dict


    def perturb_inner(self, x, y, targeted=False, x_init=None):
        l_f = self._get_loss_f(x, y, targeted, 'none')

        pert = initialize_perturbation(x, self.eps, self.norm, x_init, self.init_noise_generator)

        pert.requires_grad_(True)

        with torch.enable_grad():
            p_data = x + pert
            out = self.model(p_data)
            loss_expanded = l_f(p_data, out)
            loss = loss_expanded.mean()
            grad = torch.autograd.grad(loss, pert)[0]

        with torch.no_grad():
            pert = project_perturbation(pert - self.eps * normalize_perturbation(grad, self.norm), self.eps, self.norm)
            p_data = x + pert
            p_data = torch.clamp(p_data, 0, 1)
            final_loss = l_f(p_data, self.model(p_data))

            if self.save_trajectory:
                trajectory = torch.zeros((2,) + x.shape, device=x.device)
                trajectory[0, :] = x
                trajectory[1, :] = p_data
            else:
                trajectory = None

        return p_data, final_loss, trajectory
