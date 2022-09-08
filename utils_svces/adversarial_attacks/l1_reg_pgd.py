import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .restartattack import RestartAttack
from .utils import project_perturbation, normalize_perturbation, create_early_stopping_mask, initialize_perturbation


###################################################################################################
class L1RegularizedPGD(RestartAttack):
    def __init__(self, eps, iterations, stepsize, num_classes, reg_weight=1.0, momentum=0.9, norm='l2', loss='CrossEntropy',
                 normalize_grad=True, early_stopping=0, restarts=0, init_noise_generator=None, model=None,
                 save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.reg_weight = reg_weight
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
        dict['type'] = 'ArgminPGD'
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

        best_perts = x.new_empty(x.shape)
        best_losses = 1e13 * x.new_ones(x.shape[0])

        velocity = torch.zeros_like(x)

        #initialize perturbation
        pert = initialize_perturbation(x, self.eps, self.norm, x_init, self.init_noise_generator)
        pert_plus =  torch.zeros_like(pert)
        pert_minus =  torch.zeros_like(pert)

        pert_plus[pert > 0] = pert[pert > 0]
        pert_minus[pert < 0] = pert[pert < 0].abs()

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations + 1):
            pert_plus.requires_grad_(True)
            pert_minus.requires_grad_(True)
            pert = pert_plus - pert_minus
            with torch.enable_grad():
                p_data = x + pert
                out = self.model(p_data)
                main_loss_expanded = l_f(p_data, out)
                l1_reg = torch.sum(pert_plus + pert_minus)

                new_best = loss_expanded < best_losses
                best_losses[new_best] = loss_expanded[new_best].clone().detach()
                best_perts[new_best, :] = pert[new_best, :].clone().detach()

                if i == self.iterations:
                    break

                if self.early_stopping > 0:
                    finished, mask = create_early_stopping_mask(out, y, self.early_stopping, targeted)
                    if finished:
                        break
                else:
                    mask = 1.

                loss = torch.mean(loss_expanded)
                pert_plus_grad, pert_minus_grad = torch.autograd.grad(loss, [pert_plus, pert_minus])[0]

            with torch.no_grad():
                # pgd on given loss
                if self.normalize_grad:
                    # https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                    l1_norm_gradient =  1e-10 + torch.sum(grad.abs().reshape(x.shape[0], -1), dim=1).view(-1,1,1,1)
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

        final_loss = best_losses
        p_data = (x + best_perts).detach()
        return p_data, final_loss, trajectory
