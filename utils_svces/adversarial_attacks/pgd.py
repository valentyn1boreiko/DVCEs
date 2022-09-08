import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim
from .restartattack import RestartAttack
from .utils import project_perturbation, normalize_perturbation, create_early_stopping_mask, initialize_perturbation

class PGD(RestartAttack):
    def __init__(self, eps, iterations, stepsize, num_classes, momentum=0.9, norm='inf', loss='CrossEntropy',
                 normalize_grad=True, early_stopping=0, restarts=0, init_noise_generator=None, model=None,
                 save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.norm = norm
        self.loss = loss
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_noise_generator = init_noise_generator

    def _get_trajectory_depth(self):
        return self.iterations + 1

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'PGD'
        dict['eps'] = self.eps
        dict['iterations'] = self.iterations
        dict['stepsize'] = self.stepsize
        dict['momentum'] = self.momentum
        dict['norm'] = self.norm
        if isinstance(self.loss, str):
            dict['loss'] = self.loss
        else:
            dict['loss'] = 'custom'
        dict['normalize_grad'] = self.normalize_grad
        dict['early_stopping'] = self.early_stopping
        dict['restarts'] = self.restarts
        return dict


    def perturb_inner(self, x, y, targeted=False, x_init=None):
        l_f = self._get_loss_f(x, y, targeted, 'none')

        velocity = torch.zeros_like(x)

        #initialize perturbation
        pert = initialize_perturbation(x, self.eps, self.norm, x_init, self.init_noise_generator)

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations):
            pert.requires_grad_(True)
            with torch.enable_grad():
                p_data = x + pert
                out = self.model(p_data)

                if self.early_stopping > 0:
                    finished, mask = create_early_stopping_mask(out, y, self.early_stopping, targeted)
                    if finished:
                        break
                else:
                    mask = 1.

                loss_expanded = l_f(p_data, out)
                loss = loss_expanded.mean()
                grad = torch.autograd.grad(loss, pert)[0]

            with torch.no_grad():
                # pgd on given loss
                if self.normalize_grad:
                    # https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                    l1_norm_gradient =  1e-10 + torch.sum(grad.abs().view(x.shape[0], -1), dim=1).view(-1,1,1,1)
                    velocity = self.momentum * velocity + grad / l1_norm_gradient
                    norm_velocity = normalize_perturbation(velocity, self.norm)
                else:
                    # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
                    velocity = self.momentum * velocity + grad
                    norm_velocity = velocity

                pert = pert - self.stepsize * mask * norm_velocity
                pert = project_perturbation(pert, self.eps, self.norm)
                pert = torch.clamp(x + pert, 0, 1) - x #box constraint

                if self.save_trajectory:
                    trajectory[i + 1] = x + pert

        p_data = x + pert
        return p_data, l_f(p_data, self.model(p_data)), trajectory
