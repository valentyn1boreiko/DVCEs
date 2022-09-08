# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
#
import time
import torch
import math

from blended_diffusion.utils_blended.model_normalization import ResizeWrapper, ResizeAndMeanWrapper
from functools import partial

from .adversarialattack import AdversarialAttack

def float_(x):
    try:
       return float(x)
    except:
        return x


def maxlin(x_orig, w_orig, eps, p, verbose=False):
    ''' solves the optimization problem, for x in [0, 1]^d and p > 1,
    max <w, delta> s.th. ||delta||_p <= eps, x + delta \in [0, 1]^d
    '''
    bs = x_orig.shape[0]
    small_const = 1e-22
    x = x_orig.view(bs, -1).clamp(small_const, 1. - small_const)
    w = w_orig.view(bs, -1)
    gamma = x * (w < 0.) + (1. - x) * (w > 0.)
    delta = gamma.clone()
    w = w.abs()
    ind = gamma == 0.  # gamma < small_const #gamma == 0.
    gamma_adj, w_adj = gamma.clone(), w.clone()
    gamma_adj[ind] = small_const
    w_adj[ind] = 0.
    mus = w_adj / (p * (gamma_adj ** (p - 1)))
    if verbose:
        print('mus nan in tensor', mus.isnan().any())
        # print('w is nan', w.isnan().any())
    mussorted, ind = mus.sort(dim=1)
    gammasorted, wsorted = gamma.gather(1, ind), w_adj.gather(1, ind)
    gammacum = torch.cat([torch.zeros([bs, 1], device=x.device),
                          (gammasorted ** p).cumsum(dim=1)],
                         dim=1)
    gammacum = gammacum[:, -1].unsqueeze(1) - gammacum
    gammacum.clamp_(0.)  # small_const
    wcum = (wsorted ** (p / (p - 1))).cumsum(dim=1)
    denominator = (p * mussorted) ** (p / (p - 1))
    denominator[denominator < small_const] = small_const
    mucum = torch.cat([torch.zeros([bs, 1], device=x.device),
                       wcum / denominator], dim=1)
    if verbose:
        print('mucum is nan', mucum.isnan().any())
        print('wcum is nan', wcum.isnan().any())
        print('wsorted is nan', wsorted.isnan().any())
        print('w_adj is nan', w_adj.isnan().any())
        print('w is nan', w.isnan().any())
    fs = gammacum + mucum - eps ** p
    ind = fs[:, 0] > 0.  # * (fs[-1] < 0.)
    # print(ind)
    lb = torch.zeros(bs).long()
    ub = lb + fs.shape[1]
    u = torch.arange(bs)
    for c in range(math.ceil(math.log2(fs.shape[1]))):
        a = (lb + ub) // 2
        indnew = fs[u, a] > 0.  # - 1e-6
        lb[indnew] = a[indnew].clone()
        ub[~indnew] = a[~indnew].clone()
    pmstar = wcum[u, lb - 1] / (eps ** p - gammacum[u, lb]).clip(small_const)  # wcum[u, lb]
    if verbose:
        print('pmstar is nan', pmstar.isnan().any())
        print('pmstar pow has nan', (pmstar ** (1 / p)).isnan().any())
        '''ind_test = (pmstar ** (1 / p)).view(-1) != (pmstar ** (1 / p)).view(-1)
        print(ind_test, 1 / p, '\n', pmstar.view(-1)[ind_test], '\n', (pmstar ** (1 / p)).view(-1)[ind_test])
        print(pmstar, pmstar.shape)
        print(pmstar ** (1 / p))'''
    deltamax = w ** (1 / (p - 1)) / pmstar.unsqueeze(1) ** (1 / p)  # ** (1 / (p - 1))
    if verbose:
        print('deltamax is nan', deltamax.isnan().any())
    delta[ind] = torch.min(delta[ind],  # deltamax[ind].unsqueeze(1
                           # ) * torch.ones_like(delta[ind])
                           deltamax[ind])
    #res = delta.view(bs, -1).norm(p=p, dim=1)[ind]
    return delta.view(w_orig.shape) * w_orig.sign()

class AFWAttack(AdversarialAttack):
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            model,
            num_classes,
            eps,
            n_iter=100,
            norm='l2',
            n_restarts=1,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            fw_momentum=0,
            args=None,
            Evaluator=None,
            classifier_config=None
    ):
        """
        AutoPGD implementation in PyTorch
        """
        super().__init__(loss, num_classes, model=model, save_trajectory=False)
        self.n_iter = n_iter
        self.eps = eps
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.use_rs = True
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.fw_momentum = fw_momentum

        self.imagenet_labels = classifier_config.data.class_labels
        self.args = args
        self.device = self.args.device
        evaluator = Evaluator(self.args, classifier_config, {}, None)

        self.classifier = evaluator.load_model(
            self.args.classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_1,
                                                          interpolation=self.args.interpolation_int_1)
        )
        print('temp o resize o mean wrapper on')
        self.classifier.to(self.device)
        self.classifier.eval()

        if self.args.second_classifier_type != -1:
            self.second_classifier = evaluator.load_model(
                self.args.second_classifier_type,
                prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_2,
                                   interpolation=self.args.interpolation_int_2)
            )

            # self.second_classifier = classifier
            self.second_classifier.to(self.device)
            self.second_classifier.eval()

        if isinstance(norm, str):
            if norm.lower()[0] == 'l':
                norm = norm[1:]
            self.p = float(norm)
        else:
            self.p = float(norm)

    def init_hyperparam(self, x):
        assert not self.eps is None

        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(x.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def attack_single_run(self, x, y, targeted=False):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        device = x.device

        #init
        w_randn = torch.randn(x.shape).to(device).detach()
        x_adv = x + maxlin(x, w_randn, self.eps, self.p)

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
                                 ).to(device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
                                      ).to(device)
        acc_steps = torch.zeros_like(loss_best_steps)

        minus_criterion_indiv = self._get_loss_f(x, y, targeted, 'none')
        #my adv attacks all use a loss that takes the current perturbed datapoint and the model out at that point
        #apgd maximizes, so give a minus
        def criterion_indiv(adv_data, adv_data_out):
            return -minus_criterion_indiv(adv_data, adv_data_out)

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():

                #logits = self.model(x_adv)
                #loss_indiv = criterion_indiv(x_adv, logits)
                #loss = loss_indiv.sum()

                if self.args.second_classifier_type != -1:
                    print('using cone projection, step0')
                    logits = self.classifier(
                        x_adv)
                    logits2 = self.second_classifier(
                        x_adv)

                    loss_indiv1 = criterion_indiv(x_adv, logits)
                    loss1 = loss_indiv1.sum()

                    loss_indiv2 = criterion_indiv(x_adv, logits2)
                    loss2 = loss_indiv2.sum()

                    loss_indiv = loss_indiv2
                else:
                    logits = self.classifier(x_adv)
                    loss_indiv = criterion_indiv(x_adv, logits)
                    loss = loss_indiv.sum()

            ##grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            if self.args.second_classifier_type != -1:
                grad_temp_1 = torch.autograd.grad(loss1, x_adv, retain_graph=True)[0].view(x.shape[0],
                                                                                           -1)  # .detach()
                grad_temp_2 = torch.autograd.grad(loss2, x_adv)[0].view(x.shape[0], -1)
                angles_before = torch.acos(
                    (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))
                print('angle before', angles_before)

                grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(x.shape[0], -1)
                grad_temp_1 = grad_temp_1 - (
                        (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1) ** 2)).view(x.shape[0],
                                                                                                       -1) * grad_temp_2
                grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(x.shape[0], -1)
                # cone_projection = grad_temp_1 + grad_temp_2 45 deg
                radians = torch.tensor([self.args.deg_cone_projection], device=x.device).deg2rad()
                print('angle after', radians, torch.acos(
                    (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1))))

                cone_projection = grad_temp_1 * torch.tan(radians) + grad_temp_2

                # second classifier is a non-robust one -
                # unless we are less than 45 degrees away - don't cone project
                grad_temp = grad_temp_2.clone()
                print('norms before', grad_temp_2.norm(p=2, dim=1), grad_temp.norm(p=2, dim=1),
                      cone_projection.norm(p=2, dim=1))
                for i in range(x.shape[0]):
                    if angles_before[i] <= radians:  # 15 deg #0.7854 45 deg:
                        print('not cone projecting, angle(radian)', angles_before[i])
                    else:
                        grad_temp[i] = cone_projection[i]
                        print('project', cone_projection[i].norm(p=2))
                print('norms after', grad_temp_2.norm(p=2, dim=1), grad_temp.norm(p=2, dim=1),
                      cone_projection.norm(p=2, dim=1))

                angles_after = torch.acos((grad_temp * grad_temp_2).sum(1) / (
                        (grad_temp).norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))
                print('angle after projection', angles_after)
                grad += grad_temp.view(x.shape)
            else:
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0

        loss_best = loss_indiv.detach().clone()


        alpha = 0.75  # 0.75
        #print('Using fw, alpha is', alpha)

        step_size = alpha * torch.ones([x.shape[0], *([1] * self.ndims)]).to(device).detach()
        k = self.n_iter_2 + 0
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        m_fw = 0
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()

                #print('Using fw mode, stepsize is', step_size)
                #print('Momentum fw is', self.fw_momentum)
                if i == 0:
                    m_fw = grad
                else:
                    m_fw = self.fw_momentum * m_fw + (1 - self.fw_momentum)*grad
                v = x + maxlin(x, m_fw, self.eps, self.p)
                x_adv = x_adv + step_size * (v - x_adv)

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    ##logits = self.model(x_adv)
                    ##loss_indiv = criterion_indiv(x_adv, logits)
                    ##loss = loss_indiv.sum()
                    if self.args.second_classifier_type != -1:
                        print('using cone projection, step0')
                        logits = self.classifier(
                                x_adv)
                        logits2 = self.second_classifier(
                               x_adv)

                        loss_indiv1 = criterion_indiv(x_adv, logits)
                        loss1 = loss_indiv1.sum()

                        loss_indiv2 = criterion_indiv(x_adv, logits2)
                        loss2 = loss_indiv2.sum()

                        loss_indiv = loss_indiv2
                    else:
                        logits = self.classifier(x_adv)
                        loss_indiv = criterion_indiv(x_adv, logits)
                        loss = loss_indiv.sum()

                ###grad += torch.autograd.grad(loss, [x_adv])[0].detach()

                if self.args.second_classifier_type != -1:
                    grad_temp_1 = torch.autograd.grad(loss1, x_adv, retain_graph=True)[0].view(x.shape[0],
                                                                                                 -1)  # .detach()
                    grad_temp_2 = torch.autograd.grad(loss2, x_adv)[0].view(x.shape[0], -1)
                    angles_before = torch.acos(
                        (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))
                    print('angle before', angles_before)

                    grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(x.shape[0], -1)
                    grad_temp_1 = grad_temp_1 - (
                                (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1) ** 2)).view(x.shape[0],
                                                                                                               -1) * grad_temp_2
                    grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(x.shape[0], -1)
                    # cone_projection = grad_temp_1 + grad_temp_2 45 deg
                    radians = torch.tensor([self.args.deg_cone_projection], device=x.device).deg2rad()
                    print('angle after', radians, torch.acos(
                        (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1))))

                    cone_projection = grad_temp_1 * torch.tan(radians) + grad_temp_2

                    # second classifier is a non-robust one -
                    # unless we are less than 45 degrees away - don't cone project
                    grad_temp = grad_temp_2.clone()
                    print('norms before', grad_temp_2.norm(p=2, dim=1), grad_temp.norm(p=2, dim=1),
                          cone_projection.norm(p=2, dim=1))
                    for i in range(x.shape[0]):
                        if angles_before[i] <= radians:  # 15 deg #0.7854 45 deg:
                            print('not cone projecting, angle(radian)', angles_before[i])
                        else:
                            grad_temp[i] = cone_projection[i]
                            print('project', cone_projection[i].norm(p=2))
                    print('norms after', grad_temp_2.norm(p=2, dim=1), grad_temp.norm(p=2, dim=1),
                          cone_projection.norm(p=2, dim=1))

                    angles_after = torch.acos((grad_temp * grad_temp_2).sum(1) / (
                            (grad_temp).norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))
                    print('angle after projection', angles_after)
                    grad += grad_temp.view(x.shape)
                else:
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()



            if self.verbose:
                print('grad norms', grad.view(x_adv.shape[0], -1).norm(p=2, dim=1))
                print('targets', y)
                print('confidences', i, logits.detach().softmax(1).gather(1, y.reshape(-1, 1)))
                print('loss best', loss_best)
            if grad.view(x_adv.shape[0], -1).norm(p=2, dim=1).min() <= 1e-7:
            #    self.model.module.T*=2
                #print('shapes', self.model.module.T.shape, grad.shape, x_adv.shape)

                #if len(self.device_ids) > 1:
                #    self.model.module.T = torch.where(grad.view(x_adv.shape[0], -1).norm(p=2, dim=1).unsqueeze(1) <= 1e-7,
                #                       self.model.module.T * 2, self.model.module.T)
                #    print('doubling temp', self.model.module.T)
                #else:
                self.classifier.T = torch.where(grad.view(x_adv.shape[0], -1).norm(p=2, dim=1).unsqueeze(1) <= 1e-7,
                                                  self.classifier.T * 2, self.classifier.T)

                print('doubling temp', self.classifier.T)
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero(as_tuple=False).squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.

            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero(as_tuple=False).squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps, i, k,
                                                            loss_best, k3=self.thr_decr)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                            loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation,
                                               fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero(as_tuple=False).squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        n_reduced = fl_oscillation.sum()

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - self.size_decr, self.n_iter_min)

                    counter3 = 0

        return x_best, loss_best

    def perturb(self, x, y, targeted=False, x_init=None):
        #is_train = self.model.training
        #self.model.eval()

        self.init_hyperparam(x)
        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(x.device) * (-float('inf'))
        for counter in range(self.n_restarts):
            best_curr, loss_curr = self.attack_single_run(x, y, targeted)
            ind_curr = (loss_curr > loss_best).nonzero(as_tuple=False).squeeze()
            adv_best[ind_curr] = best_curr[ind_curr] + 0.
            loss_best[ind_curr] = loss_curr[ind_curr] + 0.

            if self.verbose:
                print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))

        #if is_train:
        #    self.model.train()
        #else:
        #    self.model.eval()

        return adv_best


