from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
import lpips
import math
import time

import cProfile, pstats, io
from pstats import SortKey

from tqdm import tqdm

from auto_attack.autopgd_base import APGDAttack

# ToDo: solve circular import and put the lambda-function in the common config
##from utils.config import FeatureDist_implemented
from utils_svces.FW import maxlin, LMO
##from utils_svces.FeatureDist import FeatureDist
from utils_svces.functions import n_restarts, noise_magnitute, blockPrint, comp_logical_tensors
from utils_svces.adversarial_attacks.utils import project_perturbation
##from utils_svces.adversarial_attacks.LPIPS_projection import project_onto_LPIPS_ball
##from datasets import inverse_data_transform
import utils_svces.adversarial_attacks as at

#try:
#    from adv_lib.utils.lagrangian_penalties import all_penalties
#except Exception as err:
#    print(str(err))

#try:
#    from adv_lib.attacks.augmented_lagrangian_attacks.alma import init_lr_finder
#except:
#    from adv_lib.attacks.augmented_lagrangian import init_lr_finder


try:
    penalty_func = all_penalties['P2']
except Exception as err:
    print(str(err))

torch.autograd.set_detect_anomaly(True)

__all__ = ['ImageSaver', 'anneal_Langevin_dynamics',
           'anneal_Langevin_dynamics_consistent', 'anneal_Langevin_dynamics_inpainting']

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

norms_dict_torch = {'l1': 1,
                    'l2': 2,
                    'l_inf': np.float('inf'),
                    'LPIPS': None,
                    'FeatureDist': 'FeatureDist',
                    'MS_SSIM_L1': 'MS_SSIM_L1'}


def float_(x):
    try:
        ret = float(x)
    except:
        ret = x
    return ret


get_norm = lambda norm_name: norms_dict_torch.get(norm_name, (lambda x: x)(float_(norm_name)))


class ImageSaver:
    def __init__(self, final_only, freq=None, clamp=False, **kwargs):
        self.images = []
        self.final_only = final_only
        self.clamp = clamp
        self.freq = freq
        self.k = 0
        self._buffer = None

    def append(self, image):
        self.k += 1
        if not self.final_only and self.k % self.freq == 0:
            self._add_image(image)
            self._buffer = None
        else:
            self._buffer = image

    def num_el(self):
        return len(self.images)

    def _add_image(self, image):
        if self.clamp:
            image = torch.clamp(image, 0.0, 1.0)
        self.images.append(image.to('cpu'))

    def __call__(self):
        if self._buffer is not None:
            self._add_image(self._buffer)
        return self.images


def reduce_tensor(tensor, reduction='sum'):
    if reduction == 'sum':
        return tensor.sum()
    elif reduction == 'none':
        return tensor
    else:
        raise NotImplementedError('Reduction is only supported for "sum" or "none"!')


def get_loss(kwargs_, out_, targets_, c=None, reduction='sum', x_0=None, x_new=None, first_sample=False,
             no_update=False, dist_penalty=1,
             loss_fn_LPIPS=None, mask=None):
    batch_size = x_new.shape[0]
    if kwargs_['d_config'].dataset.lower() == 'celeba':
        classifier_loss = -F.binary_cross_entropy_with_logits(out_, targets_,
                                                              reduction=reduction)
    elif kwargs_['RATIO_config'].activate or kwargs_['RATIO_config'].frank_wolfe.activate:
        # ConfDiff loss
        # -
        # classifier_loss = conf_diff_loss(out_, F.one_hot(targets_,
        #                                                 len(kwargs_['d_config'].class_labels)).float(),
        #                                  reduction=reduction)
        # loss_diverse_augment_condition = kwargs_['RATIO_config'].loss_diverse_iter and reduction=='none' and c < kwargs_['RATIO_config'].loss_diverse_iter
        classifier_loss = F.cross_entropy(out_, targets_, reduction=reduction)

    elif kwargs_['inverse_config'].activate:
        # penalty = torch.clamp(kwargs_['inverse_config'].inverse_mode_threshold_probs - torch.softmax(out_, dim=1).gather(1, targets_.reshape(-1, 1)),
        #                      min=0)
        # differentiable max(x, y)
        # x = kwargs_['inverse_config'].inverse_mode_threshold_probs - torch.softmax(out_, dim=1).gather(1, targets_.reshape(-1, 1))
        y = 0
        eps = 1e-12
        # penalty = 0.5 * (x + y + (x**2 + eps).sqrt())
        # penalty = torch.clamp(x, min=y)**2

        # constraint = torch.log(
        #    kwargs_['inverse_config'].inverse_mode_threshold_probs
        #    / (torch.softmax(out_, dim=1).gather(1,targets_.reshape(-1,1)))
        #              ).squeeze()

        constraint = (kwargs_['inverse_config'].inverse_mode_threshold_probs - torch.softmax(out_, dim=1)
                      .gather(1, targets_.reshape(-1, 1)).squeeze()).squeeze()

        """
        constraint_1 = x_new - 1
        constraint_0 = -x_new

        is_0 = constraint_0 <= 0
        is_1 = constraint_1 <= 0
        """

        ## is_adv = constraint < 0
        is_adv = constraint <= 0

        kwargs_['inverse_config'].is_adv = is_adv.clone()
        """
        print('is 0', is_0.all(), is_0)
        print('constraint 1', constraint_1.reshape(batch_size, -1).sum(1))
        print('constraint 0', constraint_0.reshape(batch_size, -1).sum(1))
        print('is 1', is_1.all(), is_1)
        """
        print('is adv', kwargs_['inverse_config'].is_adv)
        if first_sample:
            kwargs_['inverse_config'].prev_constraint = constraint.detach()
            """
            kwargs_['inverse_config'].prev_constraint_0 = constraint_0.detach()
            kwargs_['inverse_config'].prev_constraint_1 = constraint_1.detach()
            """

        improved_constraint = 0

        if kwargs_['inverse_config'].type == 'not_dynamic_penalty':

            # ToDo: write a function
            if kwargs_['inverse_config'].norm == 'l2':

                ##print('out_0_grad', kwargs_['inverse_config'].out_0_grad)
                ##print('out_temp_grad', kwargs_['inverse_config'].out_temp_grad)

                minimization_norm = reduce_tensor(
                    -dist_penalty * (x_0 - x_new).reshape(x_0.shape[0], -1).norm(p=2, dim=1) ** 2,
                    reduction=reduction)
                # (diff.reshape(diff.shape[0], -1)**2).sum(dim=1)
                ##reduce_tensor(
                ##    0.01 * dist_penalty * (x_0 - x_new).reshape((x_0 - x_new).shape[0], -1).norm(p=1, dim=1),
                ##    reduction=reduction) +
            elif kwargs_['inverse_config'].norm == 'l1':
                minimization_norm = reduce_tensor(
                    -dist_penalty * (x_0 - x_new).reshape((x_0 - x_new).shape[0], -1).norm(p=1, dim=1),
                    reduction=reduction)
                ##reduce_tensor(
                ##dist_penalty * (x_0 - x_new).reshape((x_0 - x_new).shape[0], -1).norm(p=2, dim=1) ** 2,
                ##reduction=reduction) +

            elif kwargs_['inverse_config'].norm in ['LPIPS', 'FeatureDist', 'MS_SSIM_L1']:
                minimization_norm = reduce_tensor(-dist_penalty * loss_fn_LPIPS(x_0, x_new).squeeze(),
                                                  reduction=reduction)
            else:
                try:
                    norm = float(kwargs_['inverse_config'].norm)
                    minimization_norm = reduce_tensor(
                        -dist_penalty * (x_0 - x_new).reshape((x_0 - x_new).shape[0], -1).norm(p=norm, dim=1) ** norm,
                        reduction=reduction)
                except ValueError:
                    raise NotImplementedError(
                        f'Inverse mode is not supported for {kwargs_["inverse_config"].norm} norm!')

            if c is not None and (c + 1) % kwargs_['inverse_config'].check_steps == 0 and not no_update:
                improved_constraint = (constraint.detach() <= kwargs_['inverse_config'].constr_improvement_rate *
                                       kwargs_['inverse_config'].prev_constraint)
                kwargs_['inverse_config'].penalty_multiplier = torch.where(
                    # ~(kwargs_['inverse_config'].adv_found | improved_constraint),
                    ~(is_adv | improved_constraint),
                    kwargs_['inverse_config'].penalty_param_increase
                    * kwargs_['inverse_config'].penalty_multiplier,
                    kwargs_['inverse_config'].penalty_multiplier).clamp(1e-6, 1e20)
                kwargs_['inverse_config'].prev_constraint = constraint.detach()

                """
                improved_constraint_0 = (constraint_0.detach() <= kwargs_['inverse_config'].constr_improvement_rate *
                                       kwargs_['inverse_config'].prev_constraint_0)

                assert constraint_0.shape == kwargs_['inverse_config'].prev_constraint_0.shape, 'Shapes mismatch!'

                kwargs_['inverse_config'].penalty_multiplier_0 = torch.where(
                    #~(kwargs_['inverse_config'].adv_found | improved_constraint),
                    ~(is_0 | improved_constraint_0).reshape(batch_size, -1).all(1),
                    kwargs_['inverse_config'].penalty_param_increase
                    * kwargs_['inverse_config'].penalty_multiplier_0,
                    kwargs_['inverse_config'].penalty_multiplier_0).clamp(1e-6, 1e20)
                kwargs_['inverse_config'].prev_constraint_0 = constraint_0.detach()

                improved_constraint_1 = (constraint_1.detach() <= kwargs_['inverse_config'].constr_improvement_rate *
                                       kwargs_['inverse_config'].prev_constraint_1)

                assert constraint_1.shape == kwargs_['inverse_config'].prev_constraint_1.shape, 'Shapes mismatch!'

                kwargs_['inverse_config'].penalty_multiplier_1 = torch.where(
                    #~(kwargs_['inverse_config'].adv_found | improved_constraint),
                    ~(is_1 | improved_constraint_1).reshape(batch_size, -1).all(1),
                    kwargs_['inverse_config'].penalty_param_increase
                    * kwargs_['inverse_config'].penalty_multiplier_1,
                    kwargs_['inverse_config'].penalty_multiplier_1).clamp(1e-6, 1e20)
                kwargs_['inverse_config'].prev_constraint_1 = constraint_1.detach()
                """

            # (torch.nn.functional.relu(constraint)
            # penalty = 0.5 * (x + y + (x**2 + eps).sqrt())
            # interior penalty function:
            # -(1/constraint) * (constraint)
            # constraint.clamp_min(0)**2

            # penalty = reduce_tensor(-kwargs_['inverse_config'].penalty_multiplier * (constraint.clamp_min(0))**2, reduction=reduction)
            ##m = kwargs_['inverse_config'].out_0_grad.shape[1]
            if mask is None:
                # cos_loss = sum([cos((x_new - x_0).reshape(batch_size, -1), kwargs_['inverse_config'].out_0_grad[:, i]) for i in range(m)])
                ##cos_loss = cos((x_new - x_0).reshape(batch_size, -1), kwargs_['inverse_config'].out_0_grad)
                penalty = reduce_tensor(-kwargs_['inverse_config'].penalty_multiplier
                                        * (torch.nn.functional.relu(constraint)),
                                        reduction=reduction)  # - 1000 * cos_loss, reduction=reduction)
            else:
                # cos_loss = sum([cos((x_new - x_0).reshape(batch_size, -1), kwargs_['inverse_config'].out_0_grad[mask, i]) for i in range(m)])
                ##cos_loss = cos((x_new - x_0).reshape(batch_size, -1), kwargs_['inverse_config'].out_0_grad[mask])
                penalty = reduce_tensor(-kwargs_['inverse_config'].penalty_multiplier
                                        * (torch.nn.functional.relu(constraint)),
                                        reduction=reduction)  # - 1000 * cos_loss, reduction=reduction)

            # penalty = reduce_tensor(-kwargs_['inverse_config'].penalty_multiplier
            #                        * (0.5*(constraint + (constraint**2 + eps).sqrt())), reduction=reduction)

            """
            penalties_box_1 = reduce_tensor(-kwargs_['inverse_config'].penalty_multiplier_1
                                    * (torch.nn.functional.relu(constraint_1)).reshape(batch_size, -1).
                                            sum(1), reduction=reduction)
            penalties_box_0 = reduce_tensor(-kwargs_['inverse_config'].penalty_multiplier_0
                                            * (torch.nn.functional.relu(constraint_0)).reshape(batch_size, -1).sum(1),
                                            reduction=reduction)
            """

            classifier_loss = penalty + (
                0 if first_sample else minimization_norm)  # + penalties_box_1 + penalties_box_0

            print('improved_constraint', improved_constraint)
            print('constraint rate', kwargs_['inverse_config'].constr_improvement_rate *
                  kwargs_['inverse_config'].prev_constraint)
            print('prev constraint', kwargs_['inverse_config'].prev_constraint)
            print('constraint', constraint)
            print('penalty multi', kwargs_['inverse_config'].penalty_multiplier.shape,
                  kwargs_['inverse_config'].penalty_multiplier)
            ##print('penalty multi 0', kwargs_['inverse_config'].penalty_multiplier_0.shape,
            ##      kwargs_['inverse_config'].penalty_multiplier_0)
            ##print('penalty multi 1', kwargs_['inverse_config'].penalty_multiplier_1.shape,
            ##      kwargs_['inverse_config'].penalty_multiplier_1)
            ##print('penalty box 1', penalties_box_1)
            ##print('penalty box 0', penalties_box_0)
            print('penalty', constraint.shape, penalty.shape, penalty)
            print('probs origin', torch.softmax(out_, dim=1).gather(1, targets_.reshape(-1, 1)))
            print('target thre', kwargs_['inverse_config'].inverse_mode_threshold_probs)
            print('minimization norm', minimization_norm)
        elif kwargs_['inverse_config'].type == 'alma':
            # penalty = (x.clamp_min(y) ** 2).squeeze()
            # penalty from https://github.com/jeromerony/adversarial-library/blob/main/adv_lib/attacks/augmented_lagrangian_attacks/alma.py
            penalty = penalty_func(constraint.clone(), kwargs_['inverse_config'].penalty_multiplier,
                                   kwargs_['inverse_config'].lagrange_multiplier)

            print('probs penalty', constraint.shape, kwargs_['inverse_config'].penalty_multiplier.shape, penalty.shape,
                  penalty)
            print('probs check', constraint)
            print('probs check penalty mutli', kwargs_['inverse_config'].penalty_multiplier)
            print('probs check lagrangian multi', kwargs_['inverse_config'].lagrange_multiplier)
            print('prob origin', torch.softmax(out_, dim=1).gather(1, targets_.reshape(-1, 1)))
            if kwargs_['inverse_config'].norm == 'l2':
                minimization_norm = reduce_tensor(
                    -dist_penalty * (x_0 - x_new).reshape((x_0 - x_new).shape[0], -1).norm(p=2, dim=1) ** 2,
                    reduction=reduction)  # (diff.reshape(diff.shape[0], -1)**2).sum(dim=1)
            elif kwargs_['inverse_config'].norm == 'l1':
                minimization_norm = 0
            elif kwargs_['inverse_config'].norm == 'LPIPS':
                minimization_norm = reduce_tensor(-dist_penalty * loss_fn_LPIPS.forward(x_0, x_new).squeeze(),
                                                  reduction=reduction)
            else:
                raise NotImplementedError('Inverse mode is only supported for l2 norm!')
            # print('minimization norm', minimization_norm)
            if first_sample:
                classifier_loss = reduce_tensor(-penalty,
                                                reduction=reduction)
                # print('loss first', classifier_loss.shape, classifier_loss)
            else:

                if c is not None and (c + 1) % kwargs_['inverse_config'].check_steps == 0 and not no_update:
                    improved_constraint = (constraint.detach() <= kwargs_['inverse_config'].constr_improvement_rate *
                                           kwargs_['inverse_config'].prev_constraint)
                    kwargs_['inverse_config'].penalty_multiplier = torch.where(
                        ~(kwargs_['inverse_config'].adv_found | improved_constraint),
                        kwargs_['inverse_config'].penalty_param_increase
                        * kwargs_['inverse_config'].penalty_multiplier,
                        kwargs_['inverse_config'].penalty_multiplier).clamp(1e-6, 1e6)
                    kwargs_['inverse_config'].prev_constraint = constraint.detach()

                if c is not None and not no_update:
                    new_lagrage_mult = torch.autograd.grad(penalty.sum(), constraint, retain_graph=True)[0]
                    kwargs_['inverse_config'].lagrange_multiplier. \
                        mul_(kwargs_['inverse_config'].ema_weight). \
                        add_(new_lagrage_mult, alpha=1 - kwargs_['inverse_config'].ema_weight). \
                        clamp_(1e-6, 1e6)

                penalty_final = penalty_func(constraint, kwargs_['inverse_config'].penalty_multiplier,
                                             kwargs_['inverse_config'].lagrange_multiplier)

                classifier_loss = reduce_tensor(-penalty_final,
                                                reduction=reduction) + minimization_norm

                # print('loss else',reduce_tensor(-penalty_final,
                #                                reduction=reduction).shape, minimization_norm.shape,  classifier_loss.shape, classifier_loss)

            print('penalty multi', kwargs_['inverse_config'].penalty_multiplier)
            print('lagrange multi', kwargs_['inverse_config'].lagrange_multiplier)
    else:
        # Cross-entropy one-hot minimization
        classifier_loss = -F.cross_entropy(out_, targets_, reduction=reduction)

    # Logits maximization
    if kwargs_['logits_max']:
        classifier_loss = 0
        for i in range(out_.shape[0]):
            classifier_loss += out_[i, targets_[i]]
    if no_update:
        if kwargs_['inverse_config'].activate:
            return constraint, classifier_loss
        else:
            return None, classifier_loss
    else:
        return classifier_loss


def anneal_Langevin_dynamics(x_mod, x_0, scorenet, sigmas, nsigma, noise_first, step_lr,
                             final_only, clamp, target, conditional=False, classifier=None,
                             save_freq=1, verbose=False, **kwargs):
    if not verbose:
        blockPrint()
    logits = []
    probs = []
    probs_all = []
    objective_vals = []
    iterates_norm = []
    scores = [] if kwargs['use_generative_model'] else None
    full_grads = []

    # Add additional eval()
    classifier.eval()
    if kwargs['Ensemble_second_classifier'] is not None:
        kwargs['Ensemble_second_classifier'].eval()

    l_0, l_1, l_2, l_inf, LPIPS, FeatureDist_arr, MS_SSIM_L1_arr = [], [], [], [], [], [], []
    first_sample = True

    images = ImageSaver(final_only, save_freq, clamp)
    denoised_images = ImageSaver(final_only, save_freq, clamp)

    L = len(sigmas)
    kwargs['L'] = L

    x_mod_ = x_mod.clone()
    batch_view = lambda tensor: tensor.reshape(batch_size, *[1] * (x_mod_.ndim - 1))
    batch_size = x_mod_.shape[0]
    device = x_mod_.device
    # ToDo: reduce to 1, put in the config
    dist_penalty = torch.full((batch_size,), 1, device=device, dtype=torch.float)

    # factors for the projections
    ##factors_sin_angle = torch.linspace(0, 1, L)
    # vars for tracking the best iterate
    least_norm_value = torch.full((batch_size, 1), 1e10, device='cpu', dtype=torch.float)
    ##least_cos_value = torch.full((batch_size, 1), 1e10, device='cpu', dtype=torch.float)

    max_confidence_value = torch.full((batch_size, 1), 0, device='cpu', dtype=torch.float)
    max_confidence_value_not_reached_threshold = torch.full((batch_size, 1), 0, device='cpu', dtype=torch.float)

    max_confidence_iterate = torch.full((batch_size, 1), 0, device='cpu', dtype=torch.float)
    max_confidence_iterate_not_reached_threshold = torch.full((batch_size, 1), 0, device='cpu', dtype=torch.float)

    best_x = x_mod.clone().to('cpu')
    best_x_not_reached_threshold = x_mod.clone().to('cpu')

    x_prev = torch.zeros_like(x_mod_)
    #images._add_image(x_mod_)
    #denoised_images._add_image(x_mod_)
    images = [x_mod_.clone().detach().cpu()]
    denoised_images = [x_mod_.clone().detach().cpu()]
    grad = 0
    momentum = 0
    prev_denoised_x = torch.tensor(0, device=device)
    if type(kwargs['inverse_config'].inverse_mode_threshold_probs) != torch.Tensor:
        inverse_threshold = torch.cuda.FloatTensor([kwargs['inverse_config'].inverse_mode_threshold_probs],
                                                   device=device)
    else:
        inverse_threshold = kwargs['inverse_config'].inverse_mode_threshold_probs

    """
    # Make sure that the mean of the right skewed distribution
    if kwargs['use_generative_model']:
        from mpmath import gamma
        num_pixels = reduce(lambda x, y: x * y, kwargs['imdims'])
        if get_norm(kwargs['norm']) == 2:
            mean_chi = np.sqrt(2) * gamma((num_pixels+1)/2)/gamma(num_pixels/2)
            # ToDo: should we just leave one epsilon?
            factor = mean_chi / kwargs['epsilons'][0]
        elif get_norm(kwargs['norm']) == 1:
            mean_half_normal = math.sqrt(2/math.pi)
            factor = (num_pixels * mean_half_normal) / kwargs['epsilons'][0]
        else:
            raise ValueError('PGD + prior is only implemented for p=1,2.')

        required_first_sigma = 1 / (factor * math.sqrt((((1 / sigmas[-1]) ** 2) * step_lr * 2)))
        assert math.isclose(sigmas[0], required_first_sigma, abs_tol=1e-4), \
            f'The first sigma ({sigmas[0]}) should be close to ({required_first_sigma}) the mean of either chi (p=2) or the sum of half-normal distribution (p=1)'
    """
    if kwargs['d_config'].dataset.lower() == 'oct':
        kwargs['target_classes'] = None
    else:
        pass
    target_classes = kwargs['target_classes'].to(device) if kwargs['target_classes'] is not None \
        else torch.ones(batch_size, device=device, dtype=torch.long) * 0
    kwargs['target_classes'] = target_classes
    denoised_x = x_mod_.clone()

    # first_run = True
    # dist_new_list = []

    if True:  # kwargs['norm'] in ['LPIPS']:
        # ToDo: do we really want to use vgg? Now it's because it makes pgd and inverse comparable
        loss_fn_LPIPS = lpips.LPIPS(
            net='vgg')  # if kwargs['regularizer'] or kwargs['inverse_config'].activate else 'alex')
        if torch.cuda.is_available():
            loss_fn_LPIPS.cuda()

        if kwargs['FeatureDistmodel'] is not None:
            loss_fn_FeatureDist = FeatureDist(kwargs['FeatureDistmodel'])
        else:
            # ToDo: make a more meaningful replacement in case FeatureDist is not implemented
            loss_fn_FeatureDist = lambda x, y: torch.norm((x - y).reshape(batch_size, -1), p=2, dim=1)
        ##loss_fn_MS_SSIM_L1_ = MS_SSIM_L1_LOSS()
        loss_fn_MS_SSIM_L1 = lambda x, y: torch.norm((x - y).reshape(batch_size, -1), p=2, dim=1).sum(1)##loss_fn_MS_SSIM_L1_(x, y).reshape(batch_size, -1).sum(1)

        if kwargs['norm'] == 'LPIPS':
            loss_fn = loss_fn_LPIPS
        elif kwargs['norm'] == 'FeatureDist':
            loss_fn = loss_fn_FeatureDist
        elif kwargs['norm'] == 'MS_SSIM_L1':
            loss_fn = loss_fn_MS_SSIM_L1
        else:
            loss_fn = None

        # ToDo: check, if the mean is fine
        dist_LPIPS = lambda x, y, mean=True: torch.norm((x - y).reshape(batch_size, -1), p=2, dim=1).sum(1) #lambda x, y, mean=True: loss_fn_LPIPS.forward(x, y).mean(0) if mean else loss_fn_LPIPS.forward(x,
                                                                                                                  #  y).cpu()
        if kwargs['FeatureDistmodel'] is not None:
            FeatureDist_fun = lambda x, y, mean=True: loss_fn_FeatureDist.forward(x, y).mean(
                0) if mean else loss_fn_FeatureDist.forward(x, y).cpu()
        else:
            FeatureDist_fun = lambda x, y, mean=True: loss_fn_FeatureDist(x, y).mean(
                0) if mean else loss_fn_FeatureDist(x, y).cpu()

        dist_MS_SSIM_L1 = lambda x, y, mean=True: loss_fn_MS_SSIM_L1(x, y).mean(0) if mean else loss_fn_MS_SSIM_L1(x,
                                                                                                                   y).cpu()

    # if kwargs['norm'] in ['LPIPS'] or p == 'LPIPS'
    _view_norm = lambda x, y, p=get_norm(kwargs['norm']), mean=True: torch.sqrt(dist_LPIPS(x, y, mean)) if p == 'LPIPS' \
        else torch.sqrt(FeatureDist_fun(x, y, mean)) if p == 'FeatureDist' else \
        dist_MS_SSIM_L1(x, y, mean) if p == 'MS_SSIM_L1' else \
            torch.norm((x - y).reshape(batch_size, -1), p=p, dim=1)

    norms = lambda x_new, x_start: (_view_norm(x_new, x_start, 0).unsqueeze(1),
                                    _view_norm(x_new, x_start, 1).unsqueeze(1),
                                    _view_norm(x_new, x_start, 2).unsqueeze(1),
                                    _view_norm(x_new, x_start, float('inf')).unsqueeze(1),

                                    _view_norm(x_new, x_start, 1).unsqueeze(1),
                                    _view_norm(x_new, x_start, 2).unsqueeze(1),
                                    _view_norm(x_new, x_start, float('inf')).unsqueeze(1),
                                    )#,
                                    #_view_norm(x_new, x_start, 'LPIPS', mean=False).unsqueeze(1),
                                    #_view_norm(x_new, x_start, 'FeatureDist', mean=False).unsqueeze(1),
                                    #_view_norm(x_new, x_start, 'MS_SSIM_L1', mean=False).unsqueeze(1))

    noise_reduction = 1

    # Init variables
    # ToDo: rewrite the whole code as a class and more efficiently
    kwargs['inverse_config'].penalty_multiplier = torch.full((batch_size,), kwargs[
        'inverse_config'].penalty_multiplier_init, device=device, dtype=torch.float)

    with torch.enable_grad():
        x_mod_ = x_mod_.requires_grad_(True)
        print('x_mod_', x_mod_)

        out_0 = classifier(x_mod_)  ##.detach()

        target_one_hot = F.one_hot(kwargs['target_classes'], len(kwargs['d_config'].class_labels))
        # second probable class
        out_0_grad = out_0.where(~target_one_hot.bool(),
                                 torch.tensor(-np.float('inf')).to(device)).max(dim=1).values
        ##out_0_grad = out_0[~target_one_hot.bool()].reshape(batch_size, len(kwargs['d_config'].class_labels)-1)
        ##print('second probs are', orig_confidences)

        ##out_0_grad = out_0.gather(1, target_classes.reshape(-1, 1))
        print('gathered out_0', out_0_grad.shape, out_0_grad)
        # m = out_0_grad.shape[1]
        # kwargs['inverse_config'].out_0_grad = []
        # for i_ in range(m):
        #    out_0_grad[:, i_].sum().backward()
        out_0_grad.sum().backward()
        #    print('grads', x_mod_.grad.data)
        #    kwargs['inverse_config'].out_0_grad.append(x_mod_.grad.data.detach().reshape(batch_size, -1))
        kwargs['inverse_config'].out_0_grad = x_mod_.grad.data.detach().reshape(batch_size, -1)
        kwargs['inverse_config'].out_0_grad /= kwargs['inverse_config'].out_0_grad.norm(p=2, dim=1).reshape(batch_size,
                                                                                                            1)
        # x_mod_.grad.data.zero_()
        print('out_0_grad', len(kwargs['inverse_config'].out_0_grad), kwargs['inverse_config'].out_0_grad)
        ##kwargs['inverse_config'].out_temp_grad = x_mod_.grad.data

    classifier_loss_0 = get_loss(kwargs_=kwargs, out_=out_0, targets_=target_classes,
                                 reduction='none', x_0=x_0, x_new=x_mod_, first_sample=True,
                                 loss_fn_LPIPS=loss_fn)
    objective_vals.append(torch.unsqueeze(
        classifier_loss_0.detach(),
        1).cpu())
    print('start loss 1', objective_vals)
    logits.append(out_0.gather(1, target_classes.reshape(-1, 1)).cpu())
    probs.append(torch.softmax(out_0, dim=1).gather(1, target_classes.reshape(-1, 1)).cpu())

    # Increase the threshold for the inverse mode
    # by rounding up the next decimal point of the current probability,
    # if current probability is greater than the threshold

    probs_all.append(torch.softmax(out_0, dim=1).unsqueeze(dim=2).cpu())
    l0_, l1_, l2_, l_inf_, LPIPS_, FeatureDist_, MS_SSIM_L1_ = norms(x_0, x_mod_)
    print('MS_SSIM', MS_SSIM_L1_)
    l_0.append(l0_.cpu())
    l_1.append(l1_.cpu())
    l_2.append(l2_.cpu())
    l_inf.append(l_inf_.cpu())
    LPIPS.append(LPIPS_.cpu())
    FeatureDist_arr.append(FeatureDist_.cpu())
    MS_SSIM_L1_arr.append(MS_SSIM_L1_.cpu())
    # Init variables

    # Using A-PGD, if chosen
    if (kwargs['RATIO_config'].activate or kwargs['RATIO_config'].apgd.activate or kwargs[
        'RATIO_config'].frank_wolfe.activate or kwargs['inverse_config'].activate or kwargs['use_generative_model']):
        # ToDo: currently the norm is fixed to l2 for apgd - change it for more norms
        counter_all = -1
        if kwargs['RATIO_config'].no_apgd or kwargs['inverse_config'].activate or \
                (kwargs['RATIO_config'].activate and (not kwargs['RATIO_config'].apgd.activate)) or kwargs[
            'use_generative_model']:
            restarts_outer = n_restarts(kwargs)
        else:
            restarts_outer = 1
        for counter_restarts in range(restarts_outer):
            x_mod_ = x_mod.clone()
            denoised_x = x_mod_.clone()
            print('restart number', counter_restarts)
            print('starting norm', x_mod_.reshape(batch_size, -1).norm(p=2, dim=1))
            print('starting norm denoised', denoised_x.reshape(batch_size, -1).norm(p=2, dim=1))
            if not (kwargs['use_generative_model'] or kwargs['RATIO_config'].activate or kwargs[
                'RATIO_config'].no_apgd):
                print('kwarg epss are', kwargs['epsilons'])
                #print([x - kwargs['epsilons'][0] for x in kwargs['epsilons']])
                #assert torch.all(torch.tensor([torch.tensor(kwargs['epsilons'])[0].eq(x) for x in torch.tensor(kwargs['epsilons'])])), 'For APGD we need a constant epsilon.'
                accepted_norms_apgd = ['l2', 'l1', 'linf']

                assert kwargs['norm'].lower() in accepted_norms_apgd or not kwargs[
                    'RATIO_config'].apgd.activate, f'Norm should be in {accepted_norms_apgd}'
                classifier.eval()
                print('seed is', kwargs['seed'])
                print('model is DataParallel', isinstance(classifier, torch.nn.DataParallel))
                start = time.time()
                print('norm is ', kwargs['norm'])

                apgd = APGDAttack(classifier, n_iter=len(sigmas) if not kwargs['inverse_config'].activate else kwargs[
                    'inverse_config'].init_apgd_steps,
                                  eot_iter=1, rho=kwargs['RATIO_config'].apgd.rho,
                                  seed=kwargs['seed'] + counter_restarts,
                                  # ToDo: accept not only lower l, or make clearer that only such are accepted
                                  device=device, norm=kwargs['norm'].replace('l', 'L') if (
                                ('l' in kwargs['norm'] or not kwargs['RATIO_config'].apgd.activate) and not kwargs[
                            'inverse_config'].activate) else "L2",
                                  eps=kwargs['epsilons'][0] if not kwargs['inverse_config'].activate else kwargs[
                                      'inverse_config'].init_apgd_eps,
                                  verbose=True,
                                  loss='ce-targeted-cfts-conf', #'ce-targeted-cfts',
                                  n_restarts=n_restarts(kwargs) if not kwargs['inverse_config'].activate else 1,  # 5
                                  use_fw=kwargs['RATIO_config'].frank_wolfe.activate,
                                  ODI_steps=kwargs['ODI_steps'],
                                  fw_momentum=kwargs['RATIO_config'].frank_wolfe.momentum,
                                  pgd_mode=kwargs['RATIO_config'].apgd.pgd_mode,
                                  pgd_step_size=step_lr,
                                  fw_constraint=kwargs['RATIO_config'].frank_wolfe.constraint,
                                  masks=kwargs['masks'] if kwargs['masks'][0] is not None else 1,
                                  dist_regularizer=None,#lambda *kw: -1 * loss_fn_LPIPS.forward(*kw)
                                  second_classifier=kwargs['Ensemble_second_classifier'],
                                  eps_increase_factor=kwargs['eps_increase_factor'],
                                  eps_increase_steps=kwargs['eps_increase_steps']
                                  )

                adv_curr, all_losses, all_confid, video = apgd.perturb(x_0, target_classes, best_loss=True)  # True)
                end = time.time()
                print('Elapsed time of the attack:', end - start)

                if not kwargs['inverse_config'].activate:
                    # ToDo: change naming from _0 to _final
                    out_0 = classifier(adv_curr)  # .requires_grad_(True))##.detach()
                    classifier_loss_0 = get_loss(kwargs_=kwargs, out_=out_0, targets_=target_classes,
                                                 reduction='none', x_0=x_0, x_new=adv_curr, first_sample=True,
                                                 loss_fn_LPIPS=loss_fn)

                    objective_vals.append(torch.unsqueeze(
                        classifier_loss_0.detach(),
                        1).cpu())
                    logits.append(out_0.gather(1, target_classes.reshape(-1, 1)).cpu())
                    gt_confs = torch.softmax(out_0, dim=1)
                    probs.append(gt_confs.gather(1, target_classes.reshape(-1, 1)).cpu())

                    probs_all.append(torch.softmax(out_0, dim=1).unsqueeze(dim=2).cpu())
                    l0_, l1_, l2_, l_inf_, LPIPS_, FeatureDist_, MS_SSIM_L1_ = norms(x_0, adv_curr)
                    print('l2', l2_)
                    l_0.append(l0_.cpu())
                    l_1.append(l1_.cpu())
                    l_2.append(l2_.cpu())
                    l_inf.append(l_inf_.cpu())
                    LPIPS.append(LPIPS_.cpu())
                    FeatureDist_arr.append(FeatureDist_.cpu())
                    MS_SSIM_L1_arr.append(MS_SSIM_L1_.cpu())
                    print('probs compare', probs[-1],
                          all_confid[torch.arange(all_confid.shape[0]), target_classes.reshape(-1), :].max(1))

                    print('max probs diff', (probs[-1].reshape(-1) -
                                             all_confid[torch.arange(all_confid.shape[0]), target_classes.reshape(-1),
                                             :].max(1)[0].reshape(-1)).abs().max())

                    # assert all(probs[-1].reshape(-1) == all_confid[torch.arange(all_confid.shape[0]), target_classes.reshape(-1), :].max(1)[0].reshape(-1))

                    prob_logit_obj_iter_stat = torch.cat(probs, dim=1), torch.cat(logits, dim=1), all_losses, \
                                               torch.cat(objective_vals, dim=1), all_confid, None, None

                    # prob_logit_obj_iter_stat = torch.cat(probs, dim=1), torch.cat(logits, dim=1), torch.cat(objective_vals, dim=1),\
                    #                           torch.cat(objective_vals, dim=1), torch.cat(probs_all,
                    #                                                                      dim=2), None, None

                    norms_ret = [l_0, l_1, l_2, l_inf, LPIPS, FeatureDist_arr, MS_SSIM_L1_arr]

                    if kwargs['eps_increase_steps'] is not None:
                        for im_ in video:
                            images.append(im_)
                            denoised_images.append(im_)
                        print('video mode', len(denoised_images))
                    else:
                        print('normal mode')
                        images.append(adv_curr.cpu())
                        denoised_images.append(adv_curr.cpu())
                else:
                    # init like in https://github.com/jeromerony/adversarial-library/blob/main/adv_lib/attacks/augmented_lagrangian_attacks/alma.py
                    kwargs['inverse_config'].penalty_multiplier = torch.full((batch_size,), kwargs[
                        'inverse_config'].penalty_multiplier_init, device=device, dtype=torch.float)
                    # kwargs['inverse_config'].penalty_multiplier_0 = torch.full((batch_size, ), kwargs['inverse_config'].penalty_multiplier_init, device=device, dtype=torch.float)
                    # kwargs['inverse_config'].penalty_multiplier_1 = torch.full((batch_size, ), kwargs['inverse_config'].penalty_multiplier_init, device=device, dtype=torch.float)
                    # kwargs['inverse_config'].lagrange_multiplier = torch.full((batch_size,), kwargs[
                    #    'inverse_config'].lagrange_multiplier_init, device=device, dtype=torch.float)
                    kwargs['inverse_config'].adv_found = torch.zeros_like(kwargs['inverse_config'].penalty_multiplier,
                                                                          dtype=torch.bool)
                    factor_step_armijo = torch.full((batch_size,), 1.0, device=device, dtype=torch.float)
                    step_found = torch.full_like(kwargs['inverse_config'].penalty_multiplier, L + 1)
                    # RMSProp-related constants
                    kwargs['inverse_config'].eps = float(kwargs['inverse_config'].eps)
                    kwargs['inverse_config'].square_avg = torch.full(x_0.shape,
                                                                     kwargs['inverse_config'].square_avg_init,
                                                                     device=device, dtype=torch.float)

                    print('probs before init', probs)
                    x_mod_ = adv_curr.clone()
                    denoised_x = adv_curr.clone()
                    print('probs after init',
                          classifier(denoised_x).detach().softmax(1).gather(1, target_classes.reshape(-1, 1)).cpu())

                if not kwargs['inverse_config'].activate and kwargs['target_classes'] is not None:
                    return images, denoised_images, gt_confs, prob_logit_obj_iter_stat, norms_ret, \
                           prob_logit_obj_iter_stat[0], prob_logit_obj_iter_stat[0]
            else:
                # Initialize delta
                delta = torch.zeros_like(x_mod_)
                # Add noise, if specified
                if noise_magnitute(kwargs):
                    #assert torch.all(torch.tensor([torch.tensor(kwargs['epsilons'])[0].eq(x) for x in torch.tensor(kwargs['epsilons'])])), \
                    #    'Not all epsilons are equal during the random init.'

                    apgd = APGDAttack(classifier,
                                      n_iter=len(sigmas) if not kwargs['inverse_config'].activate else kwargs[
                                          'inverse_config'].init_apgd_steps,
                                      eot_iter=1, rho=kwargs['RATIO_config'].apgd.rho,
                                      seed=kwargs['seed'] + counter_restarts,
                                      device=device, norm=kwargs['norm'].replace('l', 'L') if (
                                ('l' in kwargs['norm'] or not kwargs['RATIO_config'].apgd.activate) and not kwargs[
                            'inverse_config'].activate) else "L2",
                                      eps=kwargs['epsilons'][0] if not kwargs['inverse_config'].activate else kwargs[
                                          'inverse_config'].init_apgd_eps,
                                      verbose=True,
                                      loss='ce-targeted-cfts-conf',#'ce-targeted-cfts',
                                      n_restarts=n_restarts(kwargs) if not kwargs['inverse_config'].activate else 1,
                                      # 5
                                      use_fw=kwargs['RATIO_config'].frank_wolfe.activate,
                                      ODI_steps=kwargs['ODI_steps'],
                                      fw_momentum=kwargs['RATIO_config'].frank_wolfe.momentum,
                                      masks=kwargs['masks'] if kwargs['masks'][0] is not None else 1,
                                      eps_increase_factor=kwargs['eps_increase_factor'],
                                      eps_increase_steps=kwargs['eps_increase_steps']
                                      )

                    apgd.init_hyperparam(x_0)
                    delta = apgd.get_init(x_0)
                    # delta += dict_noises_like[noise_magnitute(kwargs)](x_mod_)
                    # delta = project_perturbation(delta, eps=kwargs['epsilons'][0], p=kwargs['norm'], center=x_0)
                if kwargs['RATIO_config'].loss_diverse_iter:

                    # classifier#.to(device)
                    if 'CLIP' in kwargs['type']:
                        pass
                    else:
                        classifier.eval()
                    with torch.enable_grad():
                        delta.requires_grad_(True)
                        out_init = classifier(x_0 + delta)
                        w = torch.rand_like(out_init)
                        for _ in range(kwargs['RATIO_config'].loss_diverse_iter):
                            delta = project_perturbation(delta + torch.autograd.grad((w * out_init).sum(), delta)[0],
                                                         eps=kwargs['epsilons'][0], p=kwargs['norm'], center=x_0)
                            out_init = classifier(x_0 + delta)
                print('delta is', delta)
                print('delta norm', delta.reshape(delta.shape[0], -1).norm(p=2, dim=1))
                x_mod_ += delta
                # Should I use denoised_x here?
                denoised_x = x_mod_.clone()
                delta_prev = delta

            print('distances init', (x_mod_ - x_0).reshape(batch_size, -1).norm(p=2, dim=1))

            # Gradient augmentation for diversification:
            # ToDo: should for the generative model only the last result be saved and
            #  only one of the last across restarts be evaluated as the best?
            # ToDo: check that the denoised iterate is saved for the generative model and check the influence of the n on FID

            for counter_inner, sigma in enumerate(tqdm(sigmas)):

                labels = torch.empty(batch_size, dtype=torch.long, device=device).fill_(counter_inner)

                # step-size depending on the case
                if kwargs['RATIO_config'].activate or kwargs['RATIO_config'].frank_wolfe.activate:

                    # ToDo: should I remove this adaptive_stepsize option?
                    if kwargs['RATIO_config'].adaptive_stepsize:
                        if counter_inner == math.ceil(kwargs['RATIO_config'].apgd.p_next * L):
                            print(counter_inner, 'equal!', kwargs['RATIO_config'].apgd.loss_values)
                            delta_max, kwargs['RATIO_config'].apgd.f_max_next = max(
                                kwargs['RATIO_config'].apgd.loss_values,
                                key=lambda x: x[1])

                            decrease_count = sum(
                                y - x > 0 for i, (x, y) in enumerate(kwargs['RATIO_config'].apgd.loss_values)
                                if kwargs['RATIO_config'].apgd.p_current <= i <= kwargs['RATIO_config'].apgd.p_next - 1)

                            condition_1 = decrease_count < kwargs['RATIO_config'].apgd.rho * \
                                          (kwargs['RATIO_config'].apgd.p_next - kwargs['RATIO_config'].apgd.p_current)
                            condition_2 = kwargs['RATIO_config'].apgd.lr_current == kwargs['RATIO_config'].apgd.lr_next \
                                          and kwargs['RATIO_config'].apgd.f_max_current == kwargs[
                                              'RATIO_config'].apgd.f_max_next

                            kwargs['RATIO_config'].apgd.f_max_current = kwargs['RATIO_config'].apgd.f_max_next
                            kwargs['RATIO_config'].apgd.p_next += \
                                max(kwargs['RATIO_config'].apgd.p_next
                                    - kwargs['RATIO_config'].apgd.p_current
                                    - kwargs['RATIO_config'].apgd.period_length_red,
                                    kwargs['RATIO_config'].apgd.min_length)
                            if condition_1 or condition_2:
                                kwargs['RATIO_config'].apgd.lr_current = kwargs['RATIO_config'].apgd.lr_next
                                kwargs['RATIO_config'].apgd.lr_next /= 2
                                delta = delta_max

                            print('conditions', condition_1, condition_2, kwargs['RATIO_config'].apgd.lr_next)
                        else:
                            pass

                        step_size = kwargs['RATIO_config'].apgd.lr_next
                    elif kwargs['RATIO_config'].frank_wolfe.activate:
                        if kwargs['RATIO_config'].frank_wolfe.lr_decay:
                            step_size = batch_view(
                                torch.full((batch_size,), step_lr / (step_lr + counter_inner), device=device))
                        else:
                            step_size = batch_view(
                                torch.full((batch_size,), step_lr, device=device))
                    else:
                        step_size = batch_view(
                            torch.full((batch_size,), step_lr,
                                       device=device))  # * (sigma / sigmas[-1]) ** 2, device=device))
                        ##step_size = batch_view(
                        ##    torch.full((batch_size,), step_lr * (sigma / sigmas[-1]) ** 2, device=device))
                        ##step_size = torch.full((batch_size,), kwargs['RATIO_config'].step_lr, device=device)   #step_lr * (sigma / sigmas[-1]) ** 2#
                elif kwargs['inverse_config'].activate and not kwargs['use_generative_model']:
                    if first_sample:
                        if kwargs['line_search_config'].type == 'armijo_momentum_prox':
                            # https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/08-prox-grad.pdf
                            step_size_init = torch.full((batch_size,), 1.0, device=device)
                            step_size = batch_view(step_size_init.clone())
                        else:
                            step_size_init = torch.full((batch_size,), kwargs['inverse_config'].lr_init, device=device)
                            step_size = batch_view(step_size_init.clone())
                else:
                    step_size = batch_view(torch.full((batch_size,), step_lr * (sigma / sigmas[-1]) ** 2,
                                                      device=device))  # 0.01) ** 2 # (sigma / sigmas[-1]) ** 2
                print('step size:', step_lr, sigma, sigmas[-1], step_size)
                # step_size_reverse = step_lr * (sigmas[-1] / sigma) ** 2

                for k in range(nsigma):
                    counter_all += 1
                    noise = torch.randn_like(
                        x_mod_) * noise_reduction  # / torch.log(torch.Tensor([3 + c])).to(device) #noise_reduction

                    if noise_first:
                        x_mod_ += int(kwargs['use_noise']) * noise * (batch_view(step_size) * 2).sqrt()
                        print((x_mod_ - x_0).reshape(batch_size, -1).norm(p=2, dim=1))
                        if kwargs['RATIO_config'].activate:
                            delta += int(kwargs['use_noise']) * noise * (batch_view(step_size) * 2).sqrt()  # * 0.01

                    if kwargs['use_generative_model']:
                        if target == 'dae':  # s(x) = uncorrupt(x)
                            grad = noise_reduction * (scorenet(x_mod_, labels) - x_mod_) / (sigma ** 2)
                        elif target == 'gaussian':  # s(x) = (uncorrupt(x) - x) / sigma
                            grad = noise_reduction * scorenet(x_mod_, labels) / sigma
                        else:
                            raise NotImplementedError()
                    else:
                        grad = 0

                    noise_reduction *= 1  # 0.94 #0.94 #0.94, 0.99, 0.84

                    if kwargs['use_generative_model']:
                        if target == 'gaussian':
                            scores.append(sigma * torch.unsqueeze(
                                grad.reshape(batch_size, -1).norm(p=2, dim=1),
                                1))

                            if kwargs['misalignment']:
                                # Calculate misalignment
                                print('Calculating misalignment')
                                eps = 1
                                steps = 10
                                stepsize = 0.1
                                num_classes = len(kwargs['d_config'].class_labels)
                                norm = 'l2'
                                loss = 'cosine'
                                model = lambda x: scorenet(x, labels).reshape(batch_size, -1)

                                att = at.ArgminPGD(eps, steps, stepsize, num_classes, norm=norm, loss=loss, momentum=0,
                                                   model=model)
                                gt_misalignment = att.perturb(x_mod_, grad.reshape(batch_size, -1),
                                                              targeted=True).detach()

                                print('Misalignment is generated!')
                                print(
                                    torch.nn.CosineSimilarity(dim=1, eps=1e-6)(gt_misalignment.reshape(batch_size, -1),
                                                                               grad.reshape(batch_size, -1)))

                                grad = gt_misalignment.clone()
                        else:
                            raise NotImplementedError()

                    if conditional:

                        with torch.enable_grad():
                            if kwargs['RATIO_config'].activate:
                                delta.requires_grad_(True)
                                x_mod_ = x_0 + delta
                                # ToDo: it is better to do like this?
                                denoised_x = x_mod_
                            # ToDo: is it better to do like this and to do denoising later in he loop cycle?
                            # denoised_x = x_mod_ + int(kwargs['use_noise']) * (sigma ** 2) * grad
                            denoised_x.requires_grad_(True)
                            # Derivative of log softmax
                            # ToDo: check the memory consumption, do we need to detach() somewhere
                            # classifier#.to(device)
                            if 'CLIP' in kwargs['type']:
                                pass
                            else:
                                classifier.eval()

                            if first_sample:
                                pass

                            # ToDo: allow setting any target classes
                            # ToDo: use the misclassified examples of Max not to have too different changes

                            if False and kwargs['inverse_config'].activate:
                                step_found.masked_fill_(
                                    (~kwargs['inverse_config'].adv_found) & kwargs['inverse_config'].is_adv,
                                    counter_inner)
                                kwargs['inverse_config'].adv_found.logical_or_(kwargs['inverse_config'].is_adv)
                                exp_decay = kwargs['inverse_config'].lr_reduction ** (
                                            (counter_inner - step_found).clamp_min(0) / (L - step_found))
                                # Bug fixed! Previously - step_size init
                                print('exp decay stepsize', exp_decay.shape, step_size.shape)
                                step_size = batch_view(step_size_init.clone()) * batch_view(exp_decay)
                                print('exp_decay, c, step_found, L, power, stepsize', exp_decay, counter_inner,
                                      step_found, L, ((counter_inner - step_found).clamp_min(0) / (L - step_found)),
                                      step_size)
                                print('is adv found', kwargs['inverse_config'].adv_found)

                            if kwargs['randomized_smoothing']:
                                classifier_loss = 0
                                for i in range(kwargs['noise_iterations']):
                                    print(i)
                                    x_noisy = denoised_x + kwargs['noise_sigma'] * torch.randn_like(denoised_x)
                                    out = classifier(x_noisy)
                                    classifier_loss += get_loss(kwargs_=kwargs, out_=out, targets_=sses,
                                                                c=counter_inner, x_0=x_0, x_new=denoised_x,
                                                                first_sample=first_sample, loss_fn_LPIPS=loss_fn)
                                classifier_loss /= kwargs['noise_iterations']
                            else:
                                # with torch.enable_grad():
                                # denoised_x = denoised_x.requires_grad_(True)
                                out = classifier(denoised_x)

                                ##print('gathered out_temp', out_temp_grad)
                                ##out_temp_grad.sum().backward(retain_graph=True)

                                ##kwargs['inverse_config'].out_temp_grad = denoised_x.grad.data
                                classifier_loss = get_loss(kwargs_=kwargs, out_=out, targets_=target_classes,
                                                           c=counter_inner, x_0=x_0, x_new=denoised_x,
                                                           first_sample=first_sample, loss_fn_LPIPS=loss_fn)
                                if kwargs['line_search_config'].type:
                                    _, classifier_loss_none_red = get_loss(kwargs_=kwargs, out_=out,
                                                                           targets_=target_classes, c=counter_inner,
                                                                           x_0=x_0,
                                                                           x_new=denoised_x, reduction='none',
                                                                           no_update=True,
                                                                           first_sample=first_sample,
                                                                           loss_fn_LPIPS=loss_fn)
                                # print('loss is', classifier_loss)

                            # Cross-entropy renormalized minimization
                            """
                            classifier_loss = 0
                            for i in range(out.shape[0]):
                                dist = torch.softmax(out[i, :].unsqueeze(0), dim=1)
                                if first_run:
                                    dist_new = torch.zeros_like(dist)
                                    max_, argtarget = dist.max().detach(), target_classes[i]

                                    ix = list(range(dist.shape[1]))
                                    ix.pop(argtarget)
                                    sum_max = dist[:, ix].sum().detach()
                                    dist_new[:, ix] = dist[:, ix]*torch.div((1-3*max_), sum_max)
                                    dist_new[:, target_classes[i]] = 3*max_
                                    dist_new_list.append(dist_new)
                                else:
                                    dist_new = dist_new_list[i]
                                classifier_loss += (dist_new * dist.log()).sum()
                            if first_run:
                                first_run = False
                            """
                            ##out_temp_grad = out.gather(1, target_classes.reshape(-1, 1))
                            ##print('input', out_temp_grad.sum())
                            ##first_derivative = torch.autograd.grad(out_temp_grad.sum(), inputs=denoised_x, create_graph=True)[0]
                            ##print('inputs', denoised_x)
                            ##print('first der', first_derivative.sum(), first_derivative)
                            ##print('out grad', kwargs['inverse_config'].out_0_grad)
                            ##sec_derr = torch.autograd.grad(first_derivative.sum(), inputs=denoised_x, create_graph=True)[0]
                            ##print('derr', sec_derr.shape, sec_derr.reshape(batch_size, -1).norm(p=2, dim=1))

                            ##vec = 2 * (torch.autograd.grad(out_temp_grad.sum(), denoised_x, retain_graph=True)[0] - kwargs['inverse_config'].out_0_grad)
                            # print('derr', torch.autograd.grad(first_derivative.sum(), denoised_x, retain_graph=True))
                            # print('derrrr', torch.autograd.grad(first_derivative.sum(), denoised_x, retain_graph=True))
                            # second_derivative_prod = torch.autograd.grad(first_derivative.sum(), denoised_x, grad_outputs=vec, retain_graph=True)[0]

                            grad_classifier = torch.autograd.grad(classifier_loss,
                                                                  delta if kwargs[
                                                                      'RATIO_config'].activate else denoised_x)[0]

                            ###grad_classifier -= 1.5*kwargs['inverse_config'].out_0_grad
                            ##orthog = (kwargs['inverse_config'].out_0_grad * (grad_classifier).reshape(batch_size,
                            ##                                                                        -1)).sum(
                            ##    1).reshape(batch_size, 1) * kwargs['inverse_config'].out_0_grad
                            # rotate to -45 deg
                            ##sqrt_ = (1 - cos((grad_classifier).reshape(batch_size, -1),
                            ##                 kwargs['inverse_config'].out_0_grad) ** 2)
                            ##print('sqrt_ is', sqrt_)
                            ##sin_angle = torch.sqrt(sqrt_.clamp_min(0)).reshape(batch_size, 1)
                            ##print('sin angle', sin_angle)
                            ##factor_sin_angle = factors_sin_angle[counter_inner] ##0.1763269 ## 0.052407 ##0.017455 ## 0.1763269 ##1 / torch.sqrt(torch.tensor(3.))  # 1
                            ##update = (orthog + factor_sin_angle * sin_angle * (grad_classifier).reshape(batch_size,
                            ##                                                                          -1).norm(p=2,
                            ##                                                                                   dim=1).reshape(
                            ##    batch_size, 1) * kwargs['inverse_config'].out_0_grad).reshape(batch_size,
                            ##                                                               *kwargs['imdims'])
                            ##print('update shape', update.shape)
                            ##grad_classifier -= update
                            ##print('similarity',
                            ##      cos((grad_classifier).reshape(batch_size, -1), kwargs['inverse_config'].out_0_grad))
                            ##print('hessian diag', first_derivative.shape, first_derivative.reshape(batch_size, -1).norm(p=2, dim=1))
                            ##print('vec times', vec.shape, vec.reshape(batch_size, -1).norm(p=2, dim=1))
                            ##print('grad_classifier_grad_diff is ', second_derivative_prod.shape, second_derivative_prod.reshape(batch_size, -1).norm(p=2, dim=1))
                            ###cos_dist = 100 * torch.autograd.grad(cos((denoised_x - x_0).reshape(batch_size, -1), kwargs['inverse_config'].out_0_grad.reshape(batch_size, -1)).sum(), denoised_x)[0]
                            ###print('grad cos dist is', cos_dist)
                            ###grad_classifier -= cos_dist
                            # print('grad is', grad_classifier.reshape(batch_size, -1).norm(p=2,dim=1).unsqueeze(1))
                            if kwargs['inverse_config'].activate \
                                    and kwargs['inverse_config'].init_lr_distance is not None \
                                    and counter_inner == 0:
                                grad_norm = grad_classifier.reshape(batch_size, -1).norm(p=2, dim=1)
                                randn_grad = torch.randn_like(grad_classifier).renorm(dim=0, p=2, maxnorm=1)
                                grad_classifier = torch.where(batch_view(grad_norm <= 1e-6), randn_grad,
                                                              grad_classifier)
                                step_size = batch_view(init_lr_finder(x_mod_, grad_classifier,
                                                                      distance_function=lambda y: _view_norm(x=x_mod_,
                                                                                                             y=y),
                                                                      target_distance=kwargs[
                                                                          'inverse_config'].init_lr_distance))
                                print('init stepsize', step_size)

                            # ToDo: check it momentum is needed
                            if kwargs['inverse_config'].activate and kwargs['inverse_config'].type == 'alma':
                                # if c >= L / 2:
                                #    mask_no_grad_norm = (kwargs['inverse_config'].prev_constraint > 1e-3) & (~kwargs['inverse_config'].adv_found)
                                # Using RMSProp
                                kwargs['inverse_config'].square_avg. \
                                    mul_(kwargs['inverse_config'].alpha). \
                                    addcmul_(grad_classifier, grad_classifier, value=1 - kwargs['inverse_config'].alpha)
                                avg = kwargs['inverse_config'].square_avg.sqrt(). \
                                    add_(torch.tensor(kwargs['inverse_config'].eps))
                                grad_classifier_ = grad_classifier / avg

                                # if kwargs['inverse_config'].type == 'alma':
                                grad_classifier = grad_classifier_
                                # elif c >= L / 2:
                                #    grad_classifier = torch.where(batch_view(mask_no_grad_norm), grad_classifier_, grad_classifier)

                            before_norm_grad_classifier = grad_classifier.detach().clone()
                            if kwargs['RATIO_config'].grad_normalization and not kwargs['use_generative_model']:
                                lp_norm_name = norms_dict_torch[kwargs['RATIO_config'].grad_normalization]
                                lp_norm = batch_view(grad_classifier.reshape(batch_size, -1).norm(p=lp_norm_name,
                                                                                                  dim=1))  # .reshape(-1, 1, 1, 1)
                                lp_norm_gradient = 1e-10 + lp_norm
                                velocity = grad_classifier / lp_norm_gradient
                                # if c >= L / 2:
                                #    grad_classifier = torch.where(~batch_view(mask_no_grad_norm), normalize_perturbation(velocity), grad_classifier)
                                # else:
                                grad_classifier = normalize_perturbation(velocity)

                            if False and kwargs['inverse_config'].activate and kwargs[
                                'inverse_config'].type == 'not_dynamic_penalty':
                                # Using Adam
                                momentum = kwargs['inverse_config'].beta_1 * momentum + (
                                            1 - kwargs['inverse_config'].beta_1) * grad_classifier
                                kwargs['inverse_config'].square_avg. \
                                    mul_(kwargs['inverse_config'].beta_2). \
                                    addcmul_(grad_classifier, grad_classifier,
                                             value=1 - kwargs['inverse_config'].beta_2)
                                # Bias correction terms
                                momentum_correction = momentum / (
                                            1 - kwargs['inverse_config'].beta_1 ** (counter_inner + 1))
                                square_avg_correction = kwargs['inverse_config'].square_avg / (
                                            1 - kwargs['inverse_config'].beta_2 ** (counter_inner + 1))

                                avg = square_avg_correction.sqrt(). \
                                    add_(torch.tensor(kwargs['inverse_config'].eps))

                                grad_classifier = momentum_correction / avg

                            grad_regularizer = 0

                            if kwargs['regularizer']:
                                if kwargs['regularizer'] == 'l2':
                                    print('l2 regularization')
                                    grad_regularizer = - 2 * (x_mod_ - x_0)
                                elif kwargs['regularizer'] == 'LPIPS':
                                    with torch.enable_grad():
                                        x_mod_.requires_grad_(True)
                                        grad_regularizer = - torch.autograd.grad(dist_LPIPS(x_0, x_mod_), x_mod_)[
                                            0].detach()
                                elif kwargs['regularizer'] == 'l1':
                                    pass
                                else:
                                    raise NotImplementedError('Regularization is only supported for l2 norm!')
                            if kwargs['use_generative_model']:
                                grad_scale_norm = grad.reshape(batch_size, -1).norm(p=2, dim=1)

                                grad_classifier_norm = grad_classifier.reshape(batch_size, -1).norm(p=2, dim=1)

                                print('grad classifier, generative model', list(zip(
                                    (batch_view(grad_scale_norm / grad_classifier_norm) * grad_classifier).reshape(
                                        batch_size, -1).norm(p=2, dim=1).cpu().numpy(),
                                    grad.reshape(batch_size, -1).norm(p=2, dim=1).cpu().numpy()
                                ))
                                      )
                            grad += (batch_view(grad_scale_norm / (5 * grad_classifier_norm)) if kwargs[
                                'use_generative_model'] else 1) * grad_classifier + grad_regularizer
                            ##grad += (kwargs['grad_scale'] if kwargs['use_generative_model'] else 1)*grad_classifier + grad_regularizer  #  (kwargs['epsilons'][counter_inner])

                    if kwargs['inverse_config'].activate and (kwargs['line_search_config'].type == 'wolfe_conditions'
                                                              or kwargs['line_search_config'].type == 'armijo'):
                        factor_step_armijo = torch.where(kwargs['inverse_config'].is_adv, factor_step_armijo * 0.5,
                                                         factor_step_armijo)
                        print('factor is, is_adv is', factor_step_armijo)
                        print('is adv used is', kwargs['inverse_config'].is_adv)

                        step_size = torch.full((batch_size,), kwargs['line_search_config'].alpha_current, device=device)
                        step_size = (step_size * factor_step_armijo).clamp_min(0.1)
                        step_size_start = step_size.clone().detach().cpu()
                        print("checking wolfe conditions, stepsize is", step_size_start.max(), step_size_start.mean(),
                              step_size_start.median(), step_size)

                        if not kwargs['line_search_config'].type == 'armijo':
                            upper_bound_step_size = torch.full((batch_size,), np.float('inf'), device=device)
                            lower_bound_step_size = torch.full((batch_size,),
                                                               kwargs['line_search_config'].alpha_lower_bound,
                                                               device=device)

                        grad_squared = (grad.reshape(batch_size, -1) ** 2).sum(1)
                        # counter_ = 0

                        total_steps_tensor = (-torch.log2(0.0122 / step_size_start)).int()
                        print('total steps tensor', total_steps_tensor)
                        ##0.0097
                        total_steps = 13
                        ##pr = cProfile.Profile()
                        ##pr.enable()
                        mask_1_condition_failed_composed = torch.ones(denoised_x.shape[0], dtype=bool)
                        inverse_mode_threshold_probs_origin = kwargs[
                            'inverse_config'].inverse_mode_threshold_probs.clone()
                        inverse_mode_penalty_multiplier = kwargs['inverse_config'].penalty_multiplier.clone()
                        ##out_0_grad_ = kwargs['inverse_config'].out_0_grad.clone()
                        ##out_temp_grad_ = kwargs['inverse_config'].out_temp_grad.clone()
                        for counter_ in range(
                                total_steps):  # int(-torch.log2(0.0122/step_size_start.median())):  # ToDo: use variables from the config (100*0.5^13 ~ 0.0122)
                            counter_ += 1
                            print(f'iteration {counter_}/{total_steps}')
                            # ToDo: separate x_mod_temp and denoised_x clearly in code
                            x_mod_temp = denoised_x[mask_1_condition_failed_composed].clone() + batch_view(step_size)[
                                mask_1_condition_failed_composed] * grad[mask_1_condition_failed_composed]
                            kwargs['inverse_config'].inverse_mode_threshold_probs = inverse_mode_threshold_probs_origin[
                                mask_1_condition_failed_composed]
                            kwargs['inverse_config'].penalty_multiplier = inverse_mode_penalty_multiplier[
                                mask_1_condition_failed_composed]

                            x_mod_temp = torch.clamp(x_mod_temp, 0, 1)
                            print('x_mod_temp shape', x_mod_temp.shape)
                            # Input the sliced tensor here

                            # First Wolfe condition
                            # Also input sliced tensors here
                            ##with torch.enable_grad():
                            ##x_mod_temp = x_mod_temp.requires_grad_(True)
                            out_temp = classifier(x_mod_temp)

                            ##out_temp_grad = out_temp.gather(1, target_classes[mask_1_condition_failed_composed].reshape(-1, 1))
                            ##print('gathered out_temp', out_temp_grad)
                            ##out_temp_grad.sum().backward(retain_graph=True)

                            ##kwargs['inverse_config'].out_temp_grad = x_mod_temp.grad.data

                            ##kwargs['inverse_config'].out_0_grad = out_0_grad_[mask_1_condition_failed_composed]
                            ##kwargs['inverse_config'].out_temp_grad = out_temp_grad_[mask_1_condition_failed_composed]

                            _, classifier_loss_temp = get_loss(kwargs_=kwargs, out_=out_temp, targets_=target_classes[
                                mask_1_condition_failed_composed], c=counter_inner,
                                                               x_0=x_0[mask_1_condition_failed_composed],
                                                               x_new=x_mod_temp,
                                                               first_sample=first_sample, loss_fn_LPIPS=loss_fn,
                                                               reduction='none', no_update=True,
                                                               mask=mask_1_condition_failed_composed)

                            mask_1_condition_failed = -classifier_loss_temp > (
                                        -classifier_loss_none_red[mask_1_condition_failed_composed] + step_size[
                                    mask_1_condition_failed_composed] * kwargs['line_search_config'].beta_1 * (
                                            -grad_squared[mask_1_condition_failed_composed]))
                            mask_1_condition_failed_composed = comp_logical_tensors(mask_1_condition_failed_composed,
                                                                                    mask_1_condition_failed)
                            print('first condition', classifier_loss_temp.shape, mask_1_condition_failed.shape,
                                  mask_1_condition_failed_composed.shape, mask_1_condition_failed,
                                  mask_1_condition_failed_composed)  ##, -classifier_loss_temp - (-classifier_loss_none_red[mask_1_condition_failed] + step_size[mask_1_condition_failed] * kwargs['line_search_config'].beta_1 * (-grad_squared)[mask_1_condition_failed]))

                            if not kwargs['line_search_config'].type == 'armijo':
                                upper_bound_step_size = torch.where(mask_1_condition_failed, step_size,
                                                                    upper_bound_step_size)
                                step_size = torch.where(mask_1_condition_failed,
                                                        (step_size + lower_bound_step_size) * 0.5, step_size)
                            else:
                                step_size[mask_1_condition_failed_composed] = step_size[
                                                                                  mask_1_condition_failed_composed] * 0.5  # .unsqueeze(1)[mask_1_condition_failed, 0] = step_size.unsqueeze(1)[mask_1_condition_failed, 0] * 0.5
                                #   step_size = step_size.squeeze()

                            if not kwargs['line_search_config'].type == 'armijo':
                                # Second Wolfe condition
                                x_mod_temp = denoised_x.clone() + batch_view(step_size) * grad

                                x_mod_temp = torch.clamp(x_mod_temp, 0, 1)
                                # x_mod_temp = denoised_x.clone() + batch_view(step_size) * grad
                                with torch.enable_grad():
                                    x_mod_temp.requires_grad_(True)
                                    out_temp = classifier(x_mod_temp)
                                    _, classifier_loss_temp = get_loss(kwargs_=kwargs, out_=out_temp,
                                                                       targets_=target_classes, c=counter_inner,
                                                                       x_0=x_0, x_new=x_mod_temp,
                                                                       first_sample=first_sample, loss_fn_LPIPS=loss_fn,
                                                                       reduction='none', no_update=True)
                                    classifier_loss_temp_diff = classifier_loss_temp.sum()
                                    grad_temp = torch.autograd.grad(classifier_loss_temp_diff, x_mod_temp)[0]

                                mask_2_condition_failed = (-grad_temp.reshape(batch_size, -1) * grad.reshape(batch_size,
                                                                                                             -1)).sum(
                                    1) < -(kwargs['line_search_config'].beta_2 * grad_squared)
                                ## Strong Wolfe condition
                                # mask_2_condition_failed = (grad_temp.reshape(batch_size, -1) * grad.reshape(batch_size, -1)).sum(
                                #    1).abs() > (kwargs['line_search_config'].beta_2 * grad_squared).abs()

                                # print('second condition', (-grad_temp.reshape(batch_size, -1) * grad.reshape(batch_size, -1)).sum(1) + (kwargs['line_search_config'].beta_2 * grad_squared))
                                # print('curvatures relation', (-grad_temp.reshape(batch_size, -1) * grad.reshape(batch_size, -1)).sum(1) / -(grad_squared))
                                lower_bound_step_size = torch.where(mask_2_condition_failed, step_size,
                                                                    lower_bound_step_size)
                                # two cases
                                step_size = torch.where(
                                    mask_2_condition_failed & (upper_bound_step_size != np.float('inf')),
                                    (step_size + upper_bound_step_size) * 0.5, step_size)
                                step_size = torch.where(
                                    mask_2_condition_failed & (upper_bound_step_size == np.float('inf')),
                                    step_size * 2, step_size)
                                # print('mask 2', mask_2_condition_failed)
                                # print('lower', lower_bound_step_size)
                                # print('upper', upper_bound_step_size)

                            print('mask 1', mask_1_condition_failed, mask_1_condition_failed_composed)
                            mask_1_condition_failed_composed = mask_1_condition_failed_composed & (
                                        total_steps_tensor > counter_)
                            print('mask 1 new', counter_, mask_1_condition_failed_composed,
                                  (total_steps_tensor > counter_), total_steps_tensor)
                            print('stepsize', step_size)

                            if not kwargs['line_search_config'].type == 'armijo':
                                if ((~mask_2_condition_failed) & (~mask_1_condition_failed)).all():
                                    break
                            else:
                                if (~mask_1_condition_failed_composed).all():
                                    break
                        ##pr.disable()
                        ##s = io.StringIO()
                        ##sortby = SortKey.CUMULATIVE
                        ##ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                        ##ps.print_stats()
                        ##print(s.getvalue())
                        if not kwargs['line_search_config'].type == 'armijo':
                            if mask_2_condition_failed.any() or mask_1_condition_failed.any():
                                print('Conditions not satisfied for some!')
                        else:
                            if mask_1_condition_failed.any():
                                print('Conditions not satisfied for some!')

                        if not kwargs['line_search_config'].type == 'armijo':
                            step_size = torch.where((~mask_1_condition_failed) & (~mask_2_condition_failed), step_size,
                                                    step_size_start)
                        step_size = batch_view(step_size)  # .clamp(0.0001, 10))

                        kwargs[
                            'inverse_config'].inverse_mode_threshold_probs = inverse_mode_threshold_probs_origin.clone()
                        kwargs['inverse_config'].penalty_multiplier = inverse_mode_penalty_multiplier.clone()
                        ##kwargs['inverse_config'].out_0_grad = out_0_grad_.clone()
                        ##kwargs['inverse_config'].out_temp_grad = out_temp_grad_.clone()

                        print(f"checked wolfe conditions, step_size is", step_size)

                    elif kwargs['inverse_config'].activate and kwargs[
                        'line_search_config'].type == 'armijo_momentum_prox':
                        # https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/08-prox-grad.pdf
                        ##step_size = step_size.reshape(batch_size, )
                        step_size = torch.full((batch_size,), 1000.0,
                                               device=device)  # kwargs['line_search_config'].alpha_current, device=device)
                        print("checking armijo_momentum_prox conditions, stepsize is", step_size)
                        counter_ = 0
                        # k = c + 1
                        # factor = (k - 2) / (k + 1)
                        # v = denoised_x.clone() + factor * (denoised_x.clone() - prev_denoised_x.clone())

                        """
                        with torch.enable_grad():
                            v.requires_grad_(True)
                            out_v = classifier(v)
                            _, classifier_loss_v = get_loss(kwargs_=kwargs, out_=out_v, targets_=target_classes, c=counter_inner,
                                                            x_0=x_0, x_new=v,
                                                            first_sample=first_sample, loss_fn_LPIPS=loss_fn, reduction='none',
                                                            no_update=True)
                            classifier_loss_v_diff = classifier_loss_v.sum()
                            grad_v = torch.autograd.grad(classifier_loss_v_diff, v)[0].reshape(batch_size, -1)

                            # init x, k = 1, proximal gradient step
                            x_mod_temp = torch.clamp(denoised_x.clone() + batch_view(step_size) * grad, 0, 1)
                        """

                        while counter_ < 20:
                            counter_ += 1
                            print(f'iteration {counter_}')
                            # Accelerated proximal gradient descent - commented out

                            # assert not torch.eq(denoised_x, prev_denoised_x).all(), 'denoised and previous denoised are equal in armijo_momentum_prox!'

                            G = (denoised_x.clone() - torch.clamp(denoised_x.clone() + batch_view(step_size) * grad, 0,
                                                                  1)) / batch_view(step_size)
                            x_mod_temp = denoised_x.clone() - batch_view(step_size) * G
                            out_temp = classifier(x_mod_temp)

                            _, classifier_loss_temp = get_loss(kwargs_=kwargs, out_=out_temp, targets_=target_classes,
                                                               c=counter_inner,
                                                               x_0=x_0, x_new=x_mod_temp,
                                                               first_sample=first_sample, loss_fn_LPIPS=loss_fn,
                                                               reduction='none', no_update=True)
                            ##change = (x_mod_temp - v).reshape(batch_size, -1)
                            ##mask_1_condition_failed = -classifier_loss_temp > (-classifier_loss_v - (grad_v * change).sum(1)\
                            ##                                                   + 1/(2*step_size) * (change ** 2).sum(1))

                            mask_1_condition_failed = -classifier_loss_temp > (-classifier_loss_none_red + (
                                        grad.reshape(batch_size, -1) * G.reshape(batch_size, -1)).sum(1) \
                                                                               + (step_size / 2) * (
                                                                                           G.reshape(batch_size,
                                                                                                     -1) ** 2).sum(1))
                            print('second term', (grad.reshape(batch_size, -1) * G.reshape(batch_size, -1)).sum(1))
                            print('third term', (step_size / 2) * (G.reshape(batch_size, -1) ** 2).sum(1))
                            print('loss before', -classifier_loss_none_red)
                            print('loss after', -classifier_loss_temp)
                            print('loss diff', -classifier_loss_none_red + classifier_loss_temp)
                            print('mask 1, armijo', mask_1_condition_failed)
                            # Here we set beta = 0.5
                            step_size = torch.where(mask_1_condition_failed, step_size * 0.5, step_size)
                            print('t', step_size)

                            """
                            x_mod_temp = torch.where(mask_1_condition_failed.reshape(batch_size, -1),
                                                     torch.clamp(v.reshape(batch_size, -1) + step_size.reshape(batch_size, 1) * grad_v, 0, 1),
                                                     x_mod_temp.reshape(batch_size, -1)).reshape(batch_size, *kwargs['imdims'])
                            """

                            if (~mask_1_condition_failed).all():
                                break

                        step_size = batch_view(step_size)

                    full_grads.append(
                        before_norm_grad_classifier.detach().reshape(batch_size, -1).norm(p=2, dim=1).unsqueeze(1) if
                        kwargs['RATIO_config'].grad_normalization
                        else grad.detach().reshape(batch_size, -1).norm(p=2, dim=1).unsqueeze(1))

                    if full_grads[-1].max() < 1e-6 and images.num_el() > 1:
                        print('Done!', full_grads[-1].max(), images.num_el())
                        break
                    else:
                        print('Not done!', full_grads[-1].max(), images.num_el())

                    if first_sample:
                        m = 0  # -10000000
                        # m = grad

                    if kwargs['RATIO_config'].activate:
                        BLS = None  # armijo, Lipschitz
                        if BLS is not None:
                            delta_prev_1 = delta_prev.detach().clone()
                            delta_prev = delta.detach().clone()
                            classifier_loss_none_red = get_loss(kwargs_=kwargs, out_=out, targets_=target_classes,
                                                                reduction='none').detach()
                            if first_sample:
                                alpha = torch.ones(batch_size).to(device) * 1000  # 0.1, 1, 1000

                            delta_ = project_perturbation(delta + batch_view(alpha) * (-grad),
                                                          eps=kwargs['epsilons'][counter_inner],
                                                          # .reshape(-1, 1, 1, 1)
                                                          p=kwargs['norm'], center=x_0)

                            delta_ = torch.clamp(x_0 + delta_, 0, 1) - x_0
                            a = (before_norm_grad_classifier.reshape(batch_size, -1) * (delta_).reshape(batch_size,
                                                                                                        -1)).sum(1)
                            tau = 0.5

                        if BLS == 'armijo':
                            # ToDo: is the second term grad?
                            # From https://people.maths.ox.ac.uk/hauser/hauser_lecture2.pdf

                            beta = 0.1
                            mask = (get_loss(kwargs_=kwargs,
                                             out_=classifier(denoised_x + delta_),
                                             targets_=target_classes, reduction='none')
                                    > classifier_loss_none_red + beta * a)
                            print('a', a)
                            print('class loss none red', classifier_loss_none_red)
                            print('loss change', get_loss(kwargs_=kwargs,
                                                          out_=classifier(denoised_x + delta_),
                                                          targets_=target_classes, reduction='none'))
                            print('armijo', classifier_loss_none_red + beta * a)
                            while mask.any():
                                alpha = alpha * (~mask) + mask * alpha * tau
                                delta_ = project_perturbation(delta + alpha.reshape(-1, 1, 1, 1) * (-grad),
                                                              eps=kwargs['epsilons'][counter_inner],
                                                              p=kwargs['norm'], center=x_0)

                                delta_ = torch.clamp(x_0 + delta_, 0, 1) - x_0
                                a = (before_norm_grad_classifier.reshape(batch_size, -1) * (delta_).reshape(batch_size,
                                                                                                            -1)).sum(1)
                                mask = (get_loss(kwargs_=kwargs,
                                                 out_=classifier(x_0 + delta_),
                                                 targets_=target_classes, reduction='none')
                                        > classifier_loss_none_red + beta * a)
                                print('change', alpha, mask)
                                print('change a', a)

                            step_size = batch_view(alpha)  # .reshape(-1, 1, 1, 1)
                            print('stepsize', step_size)
                        elif BLS == 'Lipschitz':
                            # ToDo: check the bounds, if they are actually correct and we can use them
                            #  - what is the paper linked to it?
                            # https://www.cs.cmu.edu/~ggordon/10725-F12/slides/09-acceleration.pdf
                            # https://scicomp.stackexchange.com/questions/26518/line-search-bracketing-for-proximal-gradient-is-it-good-idea
                            # https://pdfs.semanticscholar.org/c924/20f001e023c693db762758f9590571256e35.pdf
                            quadratic_term = ((delta_).reshape(batch_size, -1) ** 2).sum(1)

                            mask = (get_loss(kwargs_=kwargs,
                                             out_=classifier(x_0 + delta_),
                                             targets_=target_classes, reduction='none')
                                    > classifier_loss_none_red + a + quadratic_term / (2 * alpha))
                            print('change loss', get_loss(kwargs_=kwargs,
                                                          out_=classifier(x_0 + delta_),
                                                          targets_=target_classes, reduction='none'))
                            print('quadratic approx', classifier_loss_none_red + a + quadratic_term / (2 * alpha))
                            while mask.any():
                                alpha = alpha * (~mask) + mask * alpha * tau

                                delta_ = project_perturbation(delta + batch_view(alpha) * (-grad),
                                                              # .reshape(-1, 1, 1, 1)
                                                              eps=kwargs['epsilons'][counter_inner], p=kwargs['norm'],
                                                              center=x_0)

                                delta_ = torch.clamp(x_0 + delta_, 0, 1) - x_0

                                a = (before_norm_grad_classifier.reshape(batch_size, -1) * (delta_).reshape(batch_size,
                                                                                                            -1)).sum(1)
                                quadratic_term = ((delta_).reshape(batch_size, -1) ** 2).sum(1)

                                mask = (get_loss(kwargs_=kwargs,
                                                 out_=classifier(x_0 + delta_),
                                                 targets_=target_classes, reduction='none')
                                        > classifier_loss_none_red + a + quadratic_term / (2 * alpha))

                            step_size = batch_view(alpha)  # .reshape(-1, 1, 1, 1)
                            print('stepsize', step_size)
                        # print('delta before', delta)
                        delta += batch_view(step_size) * -grad
                        # print('delta after', delta)
                        ##assert not delta.eq(delta_prev).all(), 'Delta is overwritten!'
                    elif kwargs['RATIO_config'].frank_wolfe.activate:
                        m = kwargs['RATIO_config'].frank_wolfe.momentum * m + (
                                    1 - kwargs['RATIO_config'].frank_wolfe.momentum) * grad
                        # m = kwargs['RATIO_config'].frank_wolfe.momentum * m + (1 - kwargs['RATIO_config'].frank_wolfe.momentum) * grad
                        if kwargs['RATIO_config'].frank_wolfe.constraint == 'intersection':
                            print('Using intersection')
                            v = x_0 + maxlin(x_0, -m, kwargs['epsilons'][counter_inner], p=get_norm(kwargs['norm']))
                            print('v is nan', v.isnan().any())

                        elif kwargs['RATIO_config'].frank_wolfe.constraint == 'ball':
                            v = LMO(m, x_0, kwargs['epsilons'][counter_inner], p=get_norm(kwargs['norm']))
                        else:
                            raise NotImplementedError('FW supports only ball or intersection.')
                        if kwargs['RATIO_config'].frank_wolfe.backtracking_LS.activate:
                            classifier_loss_none_red = get_loss(kwargs_=kwargs, out_=out, targets_=target_classes,
                                                                reduction='none').detach()
                            BLS = 'Lipschitz'  # Lipschitz, armijo
                            if BLS == 'armijo':
                                a = (grad.reshape(batch_size, -1) * (v - x_mod_).reshape(batch_size, -1)).sum(1)
                                # From https://people.maths.ox.ac.uk/hauser/hauser_lecture2.pdf
                                alpha = torch.ones_like(a)
                                beta = 0.1
                                tau = 0.5
                                mask = (get_loss(kwargs_=kwargs,
                                                 out_=classifier(denoised_x + batch_view(alpha) * (v - x_mod_)),
                                                 # .reshape(-1, 1, 1, 1)
                                                 targets_=target_classes, reduction='none')
                                        > classifier_loss_none_red + beta * alpha * a)
                                while mask.any():
                                    alpha = alpha * (~mask) + mask * alpha * tau
                                    mask = (get_loss(kwargs_=kwargs,
                                                     out_=classifier(denoised_x + batch_view(alpha) * (v - x_mod_)),
                                                     # .reshape(-1, 1, 1, 1)
                                                     targets_=target_classes, reduction='none')
                                            > classifier_loss_none_red + beta * alpha * a)
                                    print('change', alpha, mask)

                                step_size = batch_view(alpha)  # .reshape(-1, 1, 1, 1)
                                print('stepsize', step_size)
                            # elif BLS == 'Lipschitz'

                            else:
                                # From https://arxiv.org/pdf/1806.05123.pdf
                                if kwargs['RATIO_config'].frank_wolfe.backtracking_LS.L is None:
                                    kwargs['RATIO_config'].frank_wolfe.backtracking_LS.L = torch.ones(batch_size).to(
                                        device) * 1e-6
                                    """
                                    denoised_x_ = denoised_x + kwargs['RATIO_config'].frank_wolfe.backtracking_LS.eps * (v - x_mod_)
                                    with torch.enable_grad():
                                        denoised_x_.requires_grad_(True)
                                        out_ = classifier(denoised_x_)
                                        classifier_loss_ = get_loss(kwargs_=kwargs, out_=out_, targets_ = target_classes)
                                        # ToDo: extend estimation for grad_classifier + regularizer

                                        kwargs['RATIO_config'].frank_wolfe.backtracking_LS.L = (grad_classifier - torch.autograd.grad(classifier_loss_, denoised_x_)[0])\
                                                                                               .reshape(batch_size, -1).norm(p=2, dim=1) / (kwargs['RATIO_config'].frank_wolfe.backtracking_LS.eps * (v - x_mod_)
                                                                               .reshape(batch_size, -1).norm(p=2, dim=1))
                                        """

                                M = kwargs['RATIO_config'].frank_wolfe.backtracking_LS.L * kwargs[
                                    'RATIO_config'].frank_wolfe.backtracking_LS.eta
                                a = (-grad.reshape(batch_size, -1) * (v - x_mod_).reshape(batch_size, -1)).sum(1)
                                print('a', a.shape, a)
                                b = M * ((v - x_mod_).reshape(batch_size, -1) ** 2).sum(1)
                                print(M.shape)
                                print('b', b.shape, (v - x_mod_).reshape(batch_size, -1).shape,
                                      ((v - x_mod_).reshape(batch_size, -1) ** 2).sum(1).shape, b)
                                gamma = torch.minimum(a / b,
                                                      torch.tensor(kwargs[
                                                                       'RATIO_config'].frank_wolfe.backtracking_LS.gamma_max).to(
                                                          device))
                                print('gamma', gamma.shape, gamma)

                                # ToDo: check, if we need to put minus signs
                                # .reshape(-1, 1, 1, 1)
                                mask = (get_loss(kwargs_=kwargs,
                                                 out_=classifier(denoised_x + batch_view(gamma) * (v - x_mod_)),
                                                 targets_=target_classes, reduction='none') \
                                        > quadr_approx(gamma, M, a, (v - x_mod_).reshape(batch_size, -1),
                                                       classifier_loss_none_red))
                                while mask.any():
                                    print(mask)
                                    # .reshape(-1, 1, 1, 1)
                                    print((get_loss(kwargs_=kwargs,
                                                    out_=classifier(denoised_x + batch_view(gamma) * (v - x_mod_)),
                                                    targets_=target_classes, reduction='none')))
                                    print(quadr_approx(gamma, M, a, (v - x_mod_).reshape(batch_size, -1),
                                                       classifier_loss_none_red))
                                    M = M * (~mask) + mask * M * kwargs['RATIO_config'].frank_wolfe.backtracking_LS.tau
                                    b = b * (~mask) + mask * b * kwargs['RATIO_config'].frank_wolfe.backtracking_LS.tau
                                    gamma = gamma * (~mask) + mask * torch.min(a / b,
                                                                               torch.tensor(kwargs[
                                                                                                'RATIO_config'].frank_wolfe.backtracking_LS.gamma_max).to(
                                                                                   device))
                                    mask = (get_loss(kwargs_=kwargs,
                                                     out_=classifier(denoised_x + batch_view(gamma) * (v - x_mod_)),
                                                     # .reshape(-1, 1, 1, 1)
                                                     targets_=target_classes, reduction='none') \
                                            > quadr_approx(gamma, M, a, (v - x_mod_).reshape(batch_size, -1),
                                                           classifier_loss_none_red))
                                    print('gamma update', gamma.shape, gamma)
                                kwargs['RATIO_config'].frank_wolfe.backtracking_LS.L = M
                                print('M update', M)
                                step_size = batch_view(gamma)  # .reshape(-1, 1, 1, 1)
                        x_mod_ = x_mod_ + batch_view(step_size) * (v - x_mod_)
                    elif kwargs['inverse_config'].activate:
                        # l1_norm_gradient = 1e-10 + torch.sum(grad.abs().reshape(batch_size, -1),
                        #                                     dim=1).reshape(-1, 1, 1, 1)
                        # velocity = grad / l1_norm_gradient
                        # grad = normalize_perturbation(velocity)
                        grad_temp = grad.clone()
                        if kwargs['inverse_config'].activate and counter_inner == L - 1:
                            # Do binary search to ensure being in the feasible set
                            # ToDo: use denoised x, in case inverse problem is solved with prior

                            threshold_check = lambda x, dist_penalty: get_loss(kwargs_=kwargs, out_=classifier(x),
                                                                               targets_=target_classes, c=counter_inner,
                                                                               x_0=x_0, x_new=denoised_x,
                                                                               first_sample=first_sample,
                                                                               no_update=True,
                                                                               dist_penalty=dist_penalty,
                                                                               loss_fn_LPIPS=loss_fn)
                            x_mod_temp = denoised_x.clone() + step_size * grad
                            # x_mod_temp = best_x.clone() + step_size * grad
                            threshold, _ = threshold_check(x_mod_temp, dist_penalty)
                            mask = threshold >= 0
                            print('mask start', mask)
                            dist_penalty = (dist_penalty * (~mask) + mask * dist_penalty * 0)
                            with torch.enable_grad():
                                _, loss = threshold_check(denoised_x, dist_penalty)
                                denoised_x.requires_grad_(True)
                                grad_temp = torch.autograd.grad(loss,
                                                                delta if kwargs[
                                                                    'RATIO_config'].activate else denoised_x)[0]
                            counter = 0

                            mask_alternative = (threshold < 0) | (threshold - inverse_threshold == 0)
                            step_size = torch.where(batch_view(mask), torch.ones_like(step_size) * 1e-5,
                                                    step_size)  # *1000, step_size)
                            print('new stepsize is', step_size)
                            while mask.any():
                                # assert torch.eq(denoised_x, x_mod_).all(), 'denoised has been overwritten wrongly!'
                                counter += 1
                                if counter == 100 or mask_alternative.all():
                                    break
                                # ToDo: is it correct to increase only?
                                print('counter', counter, mask_alternative.all(), mask_alternative)
                                print('mask', mask)
                                print('dist_penalty', dist_penalty)
                                print('stepsize', step_size)
                                print('thre', threshold)
                                # ToDo: is 1e-20 a good constant?
                                # 0.5).clamp_min(1e-29)
                                # ToDo: is 1 a good constant?
                                # step_size = (step_size * batch_view(~mask) + batch_view(mask) * step_size * 1.05).clamp_max(1000)

                                # step_size = (step_size * batch_view(~mask) + batch_view(mask) * step_size * 0.9).clamp_min(1e-29)
                                step_size = (step_size * batch_view(~mask) + batch_view(
                                    mask) * step_size * 1.2).clamp_min(1e-29)

                                x_mod_temp = denoised_x.clone() + step_size * grad_temp

                                # ToDo: replace clamp with the box cosntraint in the optimization objective for the inverse problem
                                x_mod_temp = torch.clamp(x_mod_temp, 0, 1)
                                threshold, _ = threshold_check(x_mod_temp, dist_penalty)

                                mask = threshold >= 0
                                mask_alternative = (threshold < 0) | (threshold - inverse_threshold == 0)
                            print('thre final', threshold)
                            denoised_x += step_size * grad_temp
                        else:
                            # if kwargs['line_search_config'].type == 'armijo_momentum_prox':
                            #    x_mod_ = x_mod_temp.clone()
                            # else:
                            x_mod_ += step_size * grad
                            # momentum = 0.9 * momentum - step_size * grad
                            # x_mod_ -= momentum
                    else:
                        x_mod_ += step_size * grad

                    if kwargs['regularizer'] == 'l1':
                        # l1 prox operator / Soft Threshold
                        # torch.clamp as cmax
                        x_mod_ = x_0 + torch.sign(x_mod_ - x_0) * torch.clamp(
                            # ToDo: adjust the multiplier 0.1
                            torch.abs(x_mod_ - x_0) - (
                                0.1 * batch_view(dist_penalty) if kwargs['inverse_config'].norm == 'l1' else
                                kwargs['epsilons'][counter_inner]) * step_size, min=0)  # 10

                    if not (kwargs['inverse_config'].activate and counter_inner == L - 1):
                        denoised_x = x_mod_.clone() + int(kwargs['use_generative_model']) * int(kwargs['use_noise']) * (
                                sigma ** 2) * grad

                    if kwargs['project']:
                        # Projection

                        print('Before proj', kwargs['epsilons'][counter_inner - 1], kwargs['norm'],
                              'norm_peturb',
                              _view_norm(x_0 + delta if kwargs['RATIO_config'].activate else x_mod_, x_0), 'norm_proj')
                        if kwargs['norm'] not in ['LPIPS']:
                            if kwargs['RATIO_config'].activate:
                                # print('delta before proj', delta.shape, delta.dtype,  _view_norm(x_0 + delta if kwargs['RATIO_config'].activate else x_mod_, x_0))
                                # ToDo: change kwargs['epsilons'][-1] to kwargs['epsilons'][counter_inner] with the correct number of elements in epsilons
                                delta = project_perturbation(delta, eps=kwargs['epsilons'][-1], p=kwargs['norm'],
                                                             center=x_0)
                                # print('delta after proj 1', delta.shape, delta.dtype, _view_norm(x_0 + delta if kwargs['RATIO_config'].activate else x_mod_, x_0))
                                if kwargs['RATIO_config'].momentum:
                                    delta = project_perturbation(
                                        delta + kwargs['RATIO_config'].apgd.alpha * (delta - delta_prev)
                                        + (1 - kwargs['RATIO_config'].apgd.alpha) * (delta_prev - delta_prev_1),
                                        eps=kwargs['epsilons'][-1], p=kwargs['norm'], center=x_0)
                                # print('delta after proj 2', delta.shape, delta.dtype,
                                #      _view_norm(x_0 + delta if kwargs['RATIO_config'].activate else x_mod_, x_0))
                                # For CLIP comment it out
                                delta = torch.clamp(x_0 + delta, 0, 1) - x_0
                                # print('delta after proj 3', delta.shape, delta.dtype,
                                #      _view_norm(x_0 + delta if kwargs['RATIO_config'].activate else x_mod_, x_0))
                                # print('x_0', x_0)

                            else:

                                x_mod_ = x_0 + project_perturbation(x_mod_ - x_0, eps=kwargs['epsilons'][-1],
                                                                    p=kwargs['norm'], center=x_0)

                                denoised_x = x_0 + project_perturbation(denoised_x - x_0, eps=kwargs['epsilons'][-1],
                                                                        p=kwargs['norm'], center=x_0)

                        else:
                            if kwargs['RATIO_config'].activate:
                                # ToDo: check that it works with loss_fn_LPIPS, before it was dist
                                delta = project_onto_LPIPS_ball(loss_fn_LPIPS, x_0 + delta, x_0,
                                                                eps=kwargs['epsilons'][-1]) - x_0
                                if kwargs['RATIO_config'].momentum:
                                    delta = project_onto_LPIPS_ball(loss_fn_LPIPS,
                                                                    x_0 + kwargs['RATIO_config'].apgd.alpha * (
                                                                                delta - delta_prev)
                                                                    + (1 - kwargs['RATIO_config'].apgd.alpha) * (
                                                                                delta_prev - delta_prev_1),
                                                                    x_0, eps=kwargs['epsilons'][-1]) \
                                            - x_0

                                delta = torch.clamp(x_0 + delta, 0, 1) - x_0
                            else:
                                x_mod_ = project_onto_LPIPS_ball(loss_fn_LPIPS, x_mod_, x_0, eps=kwargs['epsilons'][-1])
                                denoised_x = project_onto_LPIPS_ball(loss_fn_LPIPS, denoised_x, x_0,
                                                                     eps=kwargs['epsilons'][-1])

                        # For CLIP comment it out
                        x_mod_ = torch.clamp(x_mod_, 0, 1)
                        denoised_x = torch.clamp(denoised_x, 0, 1)
                        print('After proj', 'norm_peturb',
                              _view_norm(x_0 + delta if kwargs['RATIO_config'].activate else x_mod_, x_0))
                    elif (kwargs['RATIO_config'].frank_wolfe.activate and kwargs[
                        'RATIO_config'].frank_wolfe.constraint == 'ball') \
                            or kwargs['inverse_config'].activate:
                        # ToDo: replace clamp with the box constraint in the optimization objective for the inverse problem
                        x_mod_ = torch.clamp(x_mod_, 0, 1)

                        denoised_x = torch.clamp(denoised_x, 0, 1)
                        print('similarity after',
                              cos((denoised_x - x_0).reshape(batch_size, -1), kwargs['inverse_config'].out_0_grad))

                    if kwargs['RATIO_config'].activate:
                        l0_, l1_, l2_, l_inf_, LPIPS_, FeatureDist_, MS_SSIM_L1_ = norms(x_0 + delta, x_0)
                    else:
                        l0_, l1_, l2_, l_inf_, LPIPS_, FeatureDist_, MS_SSIM_L1_ = norms(denoised_x, x_0)

                    print('MS_SSIM', MS_SSIM_L1_)

                    l_0.append(l0_.cpu())
                    l_1.append(l1_.cpu())
                    l_2.append(l2_.cpu())
                    l_inf.append(l_inf_.cpu())
                    LPIPS.append(LPIPS_.cpu())
                    FeatureDist_arr.append(FeatureDist_.cpu())
                    MS_SSIM_L1_arr.append(MS_SSIM_L1_.cpu())

                    # print('Generating sample', x_mod_)
                    if not noise_first:
                        x_mod_ += int(kwargs['use_noise']) * noise * np.sqrt(step_size * 2)
                        if kwargs['RATIO_config'].activate:
                            delta += int(kwargs['use_noise']) * noise * np.sqrt(step_size * 2)  # * 0.01

                    if not final_only or (counter_inner + 1 == L and k + 1 == nsigma):
                        images.append(x_0 + delta if kwargs['RATIO_config'].activate else x_mod_)
                        denoised_images.append(x_0 + delta if kwargs['RATIO_config'].activate else denoised_x)

                    # ToDo: check that the denoised images stay in the ball
                    iterate_denoised = x_0 + delta if kwargs['RATIO_config'].activate else denoised_x

                    out = classifier(iterate_denoised)
                    logits.append(out.gather(1, target_classes.reshape(-1, 1)).cpu())
                    probs_last_ = torch.softmax(out, dim=1).gather(1, target_classes.reshape(-1, 1)).cpu()
                    probs.append(probs_last_)
                    probs_all.append(torch.softmax(out, dim=1).unsqueeze(dim=2).cpu())

                    # update optimal iterate
                    if kwargs['inverse_config'].activate:
                        # second_to_last_norm = last_norm.clone()
                        # last_norm = new_norm.clone()
                        new_norm = _view_norm(x_0, iterate_denoised, mean=False).cpu().unsqueeze(1)
                        # try either between denoised_x - x0 and out_0_grad, or grad and ... or grad_0 in the right class
                        ##new_cos = cos((denoised_x - x_0).reshape(batch_size, -1), kwargs['inverse_config'].out_0_grad.reshape(batch_size, -1)).cpu().unsqueeze(1)

                        # ToDo: can we improve this hack for the exploding norm?
                        # if c > 1:
                        #    mask_decrease_penalty_multiplier = new_norm >= 2*second_to_last_norm
                        #    print('mask decrease', mask_decrease_penalty_multiplier)
                        #    kwargs['inverse_config'].penalty_multiplier = torch.where(mask_decrease_penalty_multiplier,
                        #                                                          kwargs['inverse_config'].penalty_multiplier / kwargs['inverse_config'].penalty_param_increase,
                        #                                                          kwargs['inverse_config'].penalty_multiplier)
                        #
                        # ToDo: can we improve this hack for the smaller norm solution?
                        # kwargs['inverse_config'].penalty_multiplier = torch.where(kwargs['inverse_config'].is_adv,
                        #                                                          kwargs['inverse_config'].penalty_multiplier / kwargs['inverse_config'].penalty_param_increase,
                        #                                                                                                                    kwargs['inverse_config'].penalty_multiplier)

                        print('new norm', new_norm)
                        print('least norm', least_norm_value)
                        print('max conf not reached', max_confidence_value_not_reached_threshold)
                        ##print('least cos', least_cos_value)
                        print('maask norm', (new_norm <= least_norm_value))
                        print('mask probs', (probs_last_ >= inverse_threshold.to('cpu').unsqueeze(1)))
                        print('prob last', probs_last_)

                        mask = (probs_last_ >= inverse_threshold.to('cpu').unsqueeze(1)) & (
                                    new_norm <= least_norm_value)  ##& (new_cos <= least_cos_value)
                        # ToDo change constant 1e4, make them unified
                        mask_highest_confidence_not_reached_threshold = (
                                                                                    probs_last_ >= max_confidence_value_not_reached_threshold) & (
                                                                                    least_norm_value >= 1e4)
                        least_norm_value = new_norm * mask + least_norm_value * (~mask)
                        ##least_cos_value = new_cos * mask + least_cos_value * (~mask)
                    elif kwargs['use_generative_model']:
                        if (counter_inner == (len(sigmas) - 1)) & (k + 1 == nsigma):
                            mask = torch.full((batch_size,), True, device='cpu').unsqueeze(0).reshape(batch_size, 1)
                            mask = mask & (probs_last_ >= max_confidence_value)
                        else:
                            pass  # mask = torch.full((batch_size,), False, device='cpu')
                        # mask = torch.full((batch_size,), (counter_inner == (len(sigmas)-1)) & (k + 1 == nsigma), device='cpu', dtype=bool).\
                        #    unsqueeze(0).reshape(batch_size, 1)

                        # mask = mask & (probs_last_ >= max_confidence_value)
                    else:
                        mask = (probs_last_ >= max_confidence_value)

                    if not kwargs['use_generative_model'] or (counter_inner == (len(sigmas) - 1)) & (k + 1 == nsigma):
                        print('mask is', counter_inner, mask.shape, mask)
                        max_confidence_iterate = (counter_all + 1) * mask + max_confidence_iterate * (~mask)

                        max_confidence_value = probs_last_ * mask + max_confidence_value * (~mask)
                        best_x = iterate_denoised.cpu().clone() * batch_view(mask) + best_x * batch_view(~mask)
                        print('max_confidence_iterate is', max_confidence_iterate)
                        print('max_confidence_value is', max_confidence_value)
                        print('probs_las_', probs_last_.shape, probs_last_)

                    if kwargs['inverse_config'].activate:
                        max_confidence_value_not_reached_threshold = probs_last_ * mask_highest_confidence_not_reached_threshold + max_confidence_value_not_reached_threshold * (
                            ~mask_highest_confidence_not_reached_threshold)
                        max_confidence_iterate_not_reached_threshold = (
                                                                                   counter_all + 1) * mask_highest_confidence_not_reached_threshold + max_confidence_iterate_not_reached_threshold * (
                                                                           ~mask_highest_confidence_not_reached_threshold)
                        best_x_not_reached_threshold = iterate_denoised.cpu().clone() * batch_view(
                            mask_highest_confidence_not_reached_threshold) + best_x_not_reached_threshold * batch_view(
                            ~mask_highest_confidence_not_reached_threshold)

                    loss_ = lambda reduction: get_loss(kwargs_=kwargs, out_=out, targets_=target_classes,
                                                       reduction=reduction, x_0=x_0, x_new=denoised_x,
                                                       first_sample=first_sample, loss_fn_LPIPS=loss_fn).detach()
                    objective_vals.append(torch.unsqueeze(
                        loss_('none'),
                        1).cpu())
                    print('objective vals', objective_vals)
                    iterates_norm.append(
                        torch.unsqueeze(((x_0 + delta if kwargs['RATIO_config'].activate else denoised_x) - x_prev)
                                        .reshape(batch_size, -1).norm(p=2, dim=1),
                                        1))
                    x_prev = (x_0 + delta if kwargs['RATIO_config'].activate else denoised_x).detach().clone()

                    if kwargs['RATIO_config'].activate and kwargs['RATIO_config'].adaptive_stepsize:
                        kwargs['RATIO_config'].apgd.loss_values.append((delta, loss_('sum')))

                    if first_sample:
                        first_sample = False

                if counter_inner % nsigma == 0:
                    print(
                        f"level: {counter_inner}/{L}, total iterations: {counter_all}, {kwargs['norm']} dist: {_view_norm(x_mod_, x_0, mean=False)}")

    if kwargs['inverse_config'].activate:
        best_x = batch_view(least_norm_value < 1e4) * best_x + batch_view(
            least_norm_value >= 1e4) * best_x_not_reached_threshold
        max_confidence_value = (least_norm_value < 1e4) * max_confidence_value + (
                    least_norm_value >= 1e4) * max_confidence_value_not_reached_threshold
        max_confidence_iterate = (least_norm_value < 1e4) * max_confidence_iterate + (
                    least_norm_value >= 1e4) * max_confidence_iterate_not_reached_threshold

    # Save the best iterate
    denoised_images.append(best_x)
    # ToDo: check that the denoised images stay in the ball
    denoised_images_ = denoised_images()
    sample = denoised_images_[-1].reshape(denoised_images_[-1].shape[0], *kwargs['imdims'])
    sample = inverse_data_transform(kwargs['d_config'], sample)
    gt_out = classifier(sample.to(device))
    gt_confs = torch.softmax(gt_out, dim=1)

    print('logits', torch.cat(logits, dim=1))
    print('probs', torch.cat(probs, dim=1))
    print('probs_all', torch.cat(probs_all, dim=2))
    print('obj before', objective_vals)
    print('obj', torch.cat(objective_vals, dim=1))
    print('iterates', torch.cat(iterates_norm, dim=1))
    prob_logit_obj_iter_stat = torch.cat(probs, dim=1), torch.cat(logits, dim=1), torch.cat(objective_vals, dim=1), \
                               torch.cat(iterates_norm, dim=1), torch.cat(probs_all,
                                                                          dim=2), max_confidence_value, max_confidence_iterate
    if kwargs['use_generative_model']:
        scores = torch.cat(scores, dim=1)
    full_grads = torch.cat(full_grads, dim=1)
    print('l1', l_1)
    print('LPIPS', LPIPS)
    norms_ret = [l_0, l_1, l_2, l_inf, LPIPS, FeatureDist_arr, MS_SSIM_L1_arr]

    if kwargs['target_classes'] is not None:
        return images(), denoised_images_, gt_confs, prob_logit_obj_iter_stat, norms_ret, scores, full_grads
    else:
        return images(), denoised_images_


@torch.no_grad()
def anneal_Langevin_dynamics_consistent(x_mod, scorenet, sigmas, nsigma, noise_first, step_lr, final_only, clamp,
                                        target, save_freq=1, **kwargs):
    if target == 'dae':
        raise NotImplementedError()
    x_mod_ = x_mod.clone()

    images = ImageSaver(final_only, save_freq, clamp)
    denoised_images = ImageSaver(final_only, save_freq, clamp)

    smallest_gamma = sigmas[-1] / sigmas[-2]
    lowerbound = sigmas[-1] ** 2 * (1 - smallest_gamma)
    higherbound = sigmas[-1] ** 2 * (1 + smallest_gamma)
    assert lowerbound < step_lr < higherbound, f"Could not satisfy {lowerbound} < {step_lr} < {higherbound}"

    L = len(sigmas)
    eta = step_lr / (sigmas[-1] ** 2)

    iter_sigmas = iter(sigmas)
    next_sigma = next(iter_sigmas)

    for c in range(L):

        c_sigma = next_sigma
        score_net = scorenet(x_mod_)  # s(x) = (uncorrupt(x) - x) / sigma_k

        laststep = c + 1 == L
        if laststep or not final_only:
            denoised_x = x_mod_ + c_sigma * score_net
            denoised_images.append(denoised_x)

        x_mod_ += eta * c_sigma * score_net
        images.append(x_mod_)

        if laststep:
            continue

        next_sigma = next(iter_sigmas)
        x_mod_ += next_sigma * compute_beta(eta, next_sigma / c_sigma) * torch.randn_like(x_mod_)

        if c % nsigma == 0:
            print(f"level: {c}/{L}")

    return images(), denoised_images()


def compute_beta(eta, gamma):
    return np.sqrt(1 - ((1 - eta) / gamma) ** 2)


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size, nsigma, noise_first, step_lr,
                                        final_only, clamp, target, save_freq=1):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """
    x_mod_ = x_mod.clone()
    batch_size = x_mod_.shape[0]
    images = ImageSaver(final_only, save_freq, clamp)
    denoised_images = ImageSaver(final_only, save_freq, clamp)

    L = len(sigmas)
    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod_.shape[1], -1, -1, -1).contiguous().reshape(-1, 3,
                                                                                                        image_size,
                                                                                                        image_size)
    x_mod_ = x_mod_.reshape(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    for c, sigma in enumerate(sigmas):
        labels = torch.empty(batch_size, dtype=torch.long, device=device).fill_(c)
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for k in range(nsigma):

            if noise_first:
                x_mod_ += noise * np.sqrt(step_size * 2)

            corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
            x_mod_[:, :, :, :cols] = corrupted_half_image

            # We want grad = (uncorrupt(x) - x) / sigma^2
            if target == 'dae':  # s(x) = uncorrupt(x)
                grad = (scorenet(x_mod_, labels) - x_mod_) / (sigma ** 2)
            elif target == 'gaussian':  # s(x) = (uncorrupt(x) - x) / sigma
                grad = scorenet(x_mod_, labels) / sigma
            else:
                raise NotImplementedError()

            if not final_only or (c + 1 == L and k + 1 == nsigma):
                denoised_x = x_mod_ + (sigma ** 2) * grad
                denoised_images.append(denoised_x)
            x_mod_ += step_size * grad
            images.append(x_mod_)

            if not noise_first:
                x_mod_ += noise * np.sqrt(step_size * 2)

        if c % nsigma == 0:
            print(f"level: {c}/{L}")

    return images(), denoised_images()


def reduce_(loss, reduction):
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError('reduction not supported')


def conf_diff_loss(out, y_oh, reduction='mean'):
    # out: density_model output
    # y_oh: targets in one hot encoding
    # confidence:
    confidences = F.softmax(out, dim=1)
    conf_real = torch.sum((confidences * y_oh), 1)
    conf_other = torch.max(confidences * (1. - y_oh) - y_oh * 1e13, 1)[0]

    diff = conf_other - conf_real

    return reduce_(diff, reduction)


def normalize_perturbation(perturbation, p=2):
    if p in ['inf', 'linf', 'Linf']:
        return perturbation.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = perturbation.shape[0]
        pert_flat = perturbation.reshape(bs, -1)
        pert_normalized = F.normalize(pert_flat, p=2, dim=1)
        return pert_normalized.reshape_as(perturbation)

    else:
        raise NotImplementedError('Normalization only supports l2 and inf norm')


def quadr_approx(gamma, M, g_t, d_t, f_t):
    return f_t - gamma * g_t + (d_t ** 2).sum(1) * (gamma ** 2 * M) / 2
