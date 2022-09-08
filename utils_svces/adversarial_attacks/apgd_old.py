import numpy as np
import time
import torch

import torch.nn as nn
import torch.nn.functional as F

from .adversarialattack import AdversarialAttack


def dlr_loss(self, x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
            x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

class APGDAttack(AdversarialAttack):
    def __init__(self, model, num_classes, eps, n_iter=100, norm='Linf', n_restarts=1, seed=0, loss='ce', eot_iter=1,
                 rho=.75, verbose=False):
        super().__init__(loss, num_classes, model=model, save_trajectory=False)
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def attack_single_run(self, x_in, y_in, targeted):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() #if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter),
                                                                                              1), max(
            int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        if self.norm in ['inf','linf','Linf']:
            t = 2 * torch.rand(x.shape).to(x.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(x.device).detach() * t / (
                t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm in ['l2','L2']:
            t = torch.randn(x.shape).to(x.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(x.device).detach() * t / (
                        (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])

        minus_criterion_indiv = self._get_loss_f(x, y, targeted, 'none')
        #my adv attacks all use a loss that takes the current perturbed datapoint and the model out at that point
        #apgd maximizes, so give a minus
        def criterion_indiv(adv_data, adv_data_out):
            return -minus_criterion_indiv(adv_data, adv_data_out)


        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(x_adv, logits)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        loss_best = loss_indiv.detach().clone()

        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(x.device).detach() * torch.Tensor(
            [2.0]).to(x.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm in ['inf', 'linf', 'Linf']:
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps),
                                  x + self.eps), 0.0, 1.0)

                elif self.norm in ['l2', 'L2']:
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(x.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(x.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(x_adv, logits)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)

            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))

            ### check step out_size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero(as_tuple=False).squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k,
                                                            loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                    fl_reduce_no_impr = (~reduced_last_check) * (
                                loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.n_iter_min)

        return x_best, loss_best

    def perturb(self, x, y, targeted=False, x_init=None):
        assert self.norm in ['inf','linf','Linf', 'l2','L2']

        is_train = self.model.training
        self.model.eval()

        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(x.device) * (-float('inf'))
        for counter in range(self.n_restarts):
            best_curr, loss_curr = self.attack_single_run(x, y, targeted)
            ind_curr = (loss_curr > loss_best).nonzero(as_tuple=False).squeeze()
            adv_best[ind_curr] = best_curr[ind_curr] + 0.
            loss_best[ind_curr] = loss_curr[ind_curr] + 0.

            if self.verbose:
                print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        return adv_best