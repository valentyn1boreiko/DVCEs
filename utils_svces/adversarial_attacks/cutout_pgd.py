import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim
from .restartattack import RestartAttack
from .utils import project_perturbation, normalize_perturbation, create_early_stopping_mask


###################################################################################################
class CutoutPGD(RestartAttack):
    def __init__(self, eps, iterations, stepsize, num_classes, mask_size=16, momentum=0.9, decay=1.0, norm='inf', loss='CrossEntropy',
                 normalize_grad=False, early_stopping=0, restarts=0, init_noise_generator=None, model=None,
                 save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.mask_size = mask_size
        self.momentum = momentum
        self.decay = decay
        self.norm = norm
        self.loss = loss
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_noise_generator = init_noise_generator

    def _get_trajectory_depth(self):
        return self.iterations + 1

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'CutoutPGD'
        dict['eps'] = self.eps
        dict['iterations'] = self.iterations
        dict['stepsize'] = self.stepsize
        dict['norm'] = self.norm
        if isinstance(self.loss, str):
            dict['loss'] = self.loss
        dict['restarts'] = self.restarts
        #config_dict['init_sigma'] = self.init_sigma
        return dict


    def perturb_inner(self, x, y, targeted=False, x_init=None):
        l_f = self._get_loss_f(x, y, targeted, 'none')

        best_perts = x.new_empty(x.shape)
        best_losses = 1e13 * x.new_ones(x.shape[0])

        velocity = torch.zeros_like(x)

        cutout_mask = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
        ys = torch.randint(x.shape[2], (x.shape[0],))
        xs = torch.randint(x.shape[3], (x.shape[0],))

        y1 = torch.clamp(ys - self.mask_size // 2, 0, x.shape[2])
        y2 = torch.clamp(ys + self.mask_size // 2, 0, x.shape[2])
        x1 = torch.clamp(xs - self.mask_size // 2, 0, x.shape[3])
        x2 = torch.clamp(xs + self.mask_size // 2, 0, x.shape[3])
        for i in range(x.shape[0]):
            cutout_mask[i, :, y1[i]:y2[i], x1[i]:x2[i]] = 1


        #initialize perturbation
        if self.init_noise_generator is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_noise_generator(x)

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations + 1):
            pert.requires_grad_(True)
            with torch.enable_grad():
                p_data = x + pert
                p_data[cutout_mask] = 0
                out = self.model(p_data)
                loss_expanded = l_f(p_data, out)

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
                grad = torch.autograd.grad(loss, pert)[0]
                grad[cutout_mask] = 0.0

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

                pert = pert - (self.decay**i) * self.stepsize * mask * norm_velocity
                #todo check order
                pert = project_perturbation(pert, self.eps, self.norm)
                pert = torch.clamp(x + pert, 0, 1) - x #box constraint

                if self.save_trajectory:
                    trajectory[i + 1] = x + pert

        final_loss = best_losses
        p_data = (x + best_perts).detach()
        p_data[cutout_mask] = 0

        return p_data, final_loss, trajectory
