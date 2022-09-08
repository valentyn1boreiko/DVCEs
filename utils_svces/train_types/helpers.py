import torch
import torch.nn.functional as F
from utils_svces.adversarial_attacks import PGD, MonotonePGD, APGDAttack, ArgminPGD, UniformNoiseGenerator,\
    NormalNoiseGenerator, L2FABAttack, LinfFABAttack, CutoutPGD, AFWAttack
#from blended_diffusion.optimization import DiffusionAttack
from utils_svces.distances import LPDistance
from torch.nn.modules.batchnorm import _BatchNorm
from utils_svces.get_config import get_config
#########

def interleave_forward(model, batches, in_parallel=True):
    # interleave ref_data to preserve batch statistics on parallel computations
    #batches are supposed to have a batch sizes that are multiples of the smallest one, eg 256 and 1024
    if in_parallel:
        min_bs = min([batch.shape[0] for batch in batches])

        bs_factors = torch.empty(len(batches), dtype=torch.long)
        bs = 0
        for i in range(len(batches)):
            bs_i = batches[i].shape[0]
            bs += bs_i
            bs_factors[i] = bs_i / min_bs
            assert (bs_i % min_bs) == 0

        subdivisions = torch.sum(bs_factors).item()

        full_size = (bs,) + batches[0].shape[1:]
        full_data_interleaved = batches[0].new_empty(full_size)
        idx = 0
        batch_idcs = []
        for i in range(len(batches)):
            batch_i_idcs = []
            for j in range(bs_factors[i].item()):
                batch_i_idcs.append(torch.arange(idx, full_size[0], subdivisions, dtype=torch.long))
                idx += 1

            batch_i_idcs_cat = torch.cat(batch_i_idcs)
            full_data_interleaved[batch_i_idcs_cat, :] = batches[i]
            batch_idcs.append(batch_i_idcs_cat)

        full_out = model(full_data_interleaved)

        batches_out = []
        for i in range(len(batches)):
            batches_out.append(full_out[batch_idcs[i], :])

        return batches_out
    else:
        full_data = torch.cat(batches)
        full_out = model(full_data)
        batches_out = []
        idx = 0
        for i in range(len(batches)):
            idx_next = idx + batches[i].shape[0]
            batches_out.append(full_out[idx:idx_next, :])
            idx = idx_next

        return batches_out


def create_attack_config(eps, steps, stepsize, norm, momentum=0.9, pgd='pgd', normalize_gradient=False, noise=None):
    if noise is None:
        attack_config = {'eps': eps, 'steps': steps, 'stepsize': stepsize, 'norm': norm, 'momentum': momentum,
                         'pgd': pgd,'normalize_gradient': normalize_gradient, 'noise': None}
    elif 'uniform' in noise:
        # format: uniform_sigma
        sigma = float(noise[8:])
        attack_config = {'eps': eps, 'steps': steps, 'stepsize': stepsize, 'norm': norm, 'momentum': momentum,
                         'pgd': pgd, 'normalize_gradient': normalize_gradient, 'noise': 'uniform', 'noise_sigma': sigma}
    elif 'normal' in noise:
        # format: normal_sigma
        sigma = float(noise[7:])
        attack_config = {'eps': eps, 'steps': steps, 'stepsize': stepsize, 'norm': norm, 'momentum': momentum,
                         'pgd': pgd, 'normalize_gradient': normalize_gradient, 'noise': 'normal', 'noise_sigma': sigma}
    else:
        raise ValueError('Noise format not supported')

    return attack_config


def get_epoch_specific_config(stages_end, stages_values, epoch):
    value = 0
    for stage_end, stage_values in zip(stages_end, stages_values):
        if epoch < stage_end:
            value = stage_values
            break
    return value

def get_adversarial_attack(config, model, att_criterion, num_classes, epoch=0, args=None, Evaluator=None):
    if isinstance(config['steps'], tuple):
        stages_end, stages_values = config['steps']
        steps = get_epoch_specific_config(stages_end, stages_values, epoch)
    else:
        steps = config['steps']

    if isinstance(config['eps'], tuple):
        stages_end, stages_values = config['eps']
        eps = get_epoch_specific_config(stages_end, stages_values, epoch)
    else:
        eps = config['eps']

    if config['noise'] is None:
        noise_generator = None
    elif config['noise'] == 'uniform':
        noise_generator = UniformNoiseGenerator(min=-config['noise_sigma'],
                                                max=config['noise_sigma'])
    elif config['noise'] == 'normal':
        noise_generator = NormalNoiseGenerator(sigma=config['noise_sigma'])
    else:
        raise ValueError('Noise format not supported')

    if config['pgd'] == 'monotone':
        adv_attack = MonotonePGD(eps, steps,
                                 config['stepsize'],num_classes,
                                 momentum=config['momentum'],
                                 norm=config['norm'],
                                 loss=att_criterion, normalize_grad=config['normalize_gradient'],
                                 model=model, init_noise_generator=noise_generator)
    elif config['pgd'] == 'pgd':
        adv_attack = PGD(eps, steps,
                         config['stepsize'], num_classes,
                         momentum=config['momentum'], norm=config['norm'],
                         loss=att_criterion, normalize_grad=config['normalize_gradient'],
                         model= model, init_noise_generator=noise_generator)
    elif config['pgd'] == 'argmin':
        adv_attack = ArgminPGD(eps, steps,
                               config['stepsize'], num_classes,
                               momentum=config['momentum'], norm=config['norm'],
                               loss=att_criterion, normalize_grad=config['normalize_gradient'],
                               model= model, init_noise_generator=noise_generator)
    elif config['pgd'] == 'apgd':
        adv_attack = APGDAttack(model, eps=eps, n_iter=steps, norm=config['norm'], loss=att_criterion)
    elif config['pgd'] == 'cutoutpgd':
        adv_attack = CutoutPGD(eps, steps,
                               config['stepsize'], num_classes,
                               momentum=config['momentum'], norm=config['norm'],
                               loss=att_criterion, normalize_grad=config['normalize_gradient'],
                               model= model, init_noise_generator=noise_generator)
    elif config['pgd'] == 'fab':
        if config['norm'] in ['inf', 'linf', 'Linf']:
            adv_attack = LinfFABAttack( model, n_restarts=1, n_iter=steps, eps=eps)
        elif config['norm'] in ['l2', 'L2']:
            adv_attack = L2FABAttack( model, n_restarts=1, n_iter=steps, eps=eps)
        else:
            raise NotImplementedError('Norm not supported')
    elif config['pgd'] == 'afw':
        classifier_config = get_config(args)
        adv_attack = AFWAttack(None, num_classes, eps, n_iter=steps, norm=config['norm'], loss=att_criterion, args=args, Evaluator=Evaluator, classifier_config=classifier_config)
    ##elif config['pgd'] == 'diffusion':
    ##    adv_attack = DiffusionAttack(args)
    else:
        raise ValueError('PGD {} not supported'.format(config['pgd']))
    return adv_attack


def get_distance(norm):
    if norm in ['inf', 'linf', 'Linf', 'LINF']:
        distance = LPDistance(p='inf')
    else:
        try:
            if isinstance(norm, str):
                if norm.lower()[0] == 'l':
                    norm = norm[1:]
                p = float(norm)
            else:
                p = float(norm)
            distance = LPDistance(p=p)
        except Exception as e:
            raise NotImplementedError('Norm not supported')

    return distance


def disable_running_stats(model):
    def _disable(module):
        if issubclass(type(module), _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if issubclass(type(module), _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
