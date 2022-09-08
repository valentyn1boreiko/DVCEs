from distutils.util import strtobool
import utils_svces.train_types.schedulers as schedulers
import utils_svces.train_types.optimizers as optimizers
import utils_svces.train_types.msda as msda
from utils_svces.datasets import *
from utils_svces.model_normalization import *
import utils_svces.train_types as tt
import os
import yaml
import argparse
import sys
import utils_svces.models.model_factory_32 as factory_32
import utils_svces.models.model_factory_224 as factory_224

models_dict = {"cifar10": Cifar10Wrapper,
               "restrictedimagenet": RestrictedImageNetWrapper,
               "cifar100": Cifar100Wrapper,
               "funduskaggle": FundusKaggleWrapper_clahe_v2_new_qual_eval_drop1, #FundusKaggleWrapper_clahe_v2_new_qual_eval, #FundusKaggleWrapper_raw_v2_new_qual_eval_artifacts_green_circles_blue_squares, #FundusKaggleWrapper_clahe_v2_new_qual_eval, #FundusKaggleWrapper_clahe_v2_new_qual_eval,#FundusKaggleWrapper_raw_clahe_v2, #FundusKaggleWrapper, #FundusKaggleWrapperBackgroundSubtracted, #FundusKaggleWrapper,
               "funduskaggle_background_sub": FundusKaggleWrapper,# FundusKaggleWrapperBackgroundSubtracted}
               "oct": OCTWrapper_first1000}
               #"funduskaggle": FundusKaggleWrapper}

evaldirs_dict = {"cifar10": "Cifar10Eval/",
                 "restrictedimagenet": "RestrictedImageNetEval/",
                 "cifar100": "Cifar100Eval/",
                 "funduskaggle": "fundusKaggleEval/"}

data_loader_dict = {"cifar10": get_CIFAR10,
                    "semi-cifar10": get_CIFAR10_ti_500k,
                    "tinyimages": get_80MTinyImages,
                    "cifar100": get_CIFAR100,
                    "restrictedimagenet": get_restrictedImageNet,
                    "openimages": get_openImages,
                    "restrictedimagenetod": get_restrictedImageNetOD,
                    "funduskaggle": get_FundusKaggle}

factory_dict = {32: factory_32,
                224: factory_224}


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parser_add_commons(parser):
    parser.add_argument('--task', type=str, default='', help='cifar10/cifar100/restrictedImageNet')
    parser.add_argument('--img_size', type=int, default=0, help='Image size')
    parser.add_argument('--num_classes', type=int, default=0, help='Number of classes')
    parser.add_argument('--class_labels', type=list, default=None, help='Class labels (usually defined in a config)')
    parser.add_argument('--model_root_dir', type=str, default='', help='Root directory for the model')
    parser.add_argument('--logs_root_dir', type=str, default='', help='Root directory for the logs')
    parser.add_argument('--net', type=str, default='', help='Resnet18, 34 or 50, WideResNet28')
    parser.add_argument('--dataset', type=str, default='', help='cifar10 or semi-cifar10')
    parser.add_argument('--od_dataset', type=str, default='',
                        help=('tinyImages or cifar100'))
    parser.add_argument('--exclude_cifar', dest='exclude_cifar', type=lambda x: bool(strtobool(x)),
                        default=True, help='whether to exclude cifar10 from tiny images')
    parser.add_argument('--background_subtraction', dest='background_subtraction', type=lambda x: bool(strtobool(x)),
                        default=False, help='whether to subtract background')
    parser.add_argument('--gpu', '--list', nargs='+', default=[0],
                        help='GPU indices, if more than 1 parallel modules will be called')
    parser.add_argument('--class_weights', default=None,
                        help='Class weights used for ce loss. For balanced problems set to 1,'
                             ' for imbalanced - accordingly')
    parser.add_argument('--clean_criterion', default='ce',
                        help='ce(crossentropy)/focal')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--bs', type=int, default=128, help='Training batch size')
    parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
    parser.add_argument('--test_epochs', type=int, default=1, help='Test frequency')
    parser.add_argument('--train_type', type=str, default='plain', help='Train type')
    parser.add_argument('--continue', dest='continue_trained', nargs=3, type=str,
                        default=None, help='Filename of density_model to load an epoch')
    parser.add_argument('--augm', type=str, default='default',
                        help=('Augmentation type (none/murat fpr fundusKaggle)'))
    parser.add_argument('--warmup_epochs', type=int,
                        default='0', help='Warmup length')
    parser.add_argument('--scheduler', type=str,
                        default='step_lr', help='LR scheduler')
    parser.add_argument('--nesterov', dest='nesterov', type=lambda x: bool(strtobool(x)),
                        default=True, help='Nesterov SGD')
    parser.add_argument('--msda', type=str,
                        default='none', help='MSDA: None, mixup or fmix')
    parser.add_argument('--msda_alpha', type=float,
                        default=1.0, help='MSDA Alpha')
    parser.add_argument('--mixed_precision', type=lambda x: bool(strtobool(x)),
                        default=False, help='Mixed precision training')

    parser.add_argument('--swa', type=str,
                        default=None, help='SWA cosine or const')
    parser.add_argument('--swa_epochs', type=int,
                        default=500, help='SWA epochs after regular training')
    parser.add_argument('--swa_cycle_length', type=int,
                        default=200,
                        help='In Cosine mode, SWA repeats epochs [SWA_END - SWA_CYCLE_LENGTH, SWA_END] for a total number of SWA_epochs')
    parser.add_argument('--swa_virtual_schedule_length', type=int,
                        default=1800,
                        help='In Cosine mode, SWA repeats epochs [SWA_END - SWA_CYCLE_LENGTH, SWA_END] for a total number of SWA_epochs')
    parser.add_argument('--swa_virtual_schedule_swa_end', type=int,
                        default=1500,
                        help='In Cosine mode, SWA repeats epochs [SWA_END - SWA_CYCLE_LENGTH, SWA_END] for a total number of SWA_epochs')
    parser.add_argument('--swa_lr', type=float,
                        default=0.025,
                        help='SWA LR')
    parser.add_argument('--swa_update_frequency', type=int,
                        default=20, help='SWA epochs after regular training')


def parser_add_config(parser):
    parser.add_argument('--config', type=str, help='Path to the config file(optional)')

    hps = parser.parse_args()
    if hps.config:
        with open(os.path.join('configs', hps.config), 'r') as f:
            config = yaml.load(f)
        new_config = dict2namespace(config)
        new_config_dict = vars(new_config)
        for key, value in vars(hps).items():
            if key in new_config and '--' + key not in sys.argv:
                setattr(hps, key, eval(new_config_dict[key]) if 'inf' in key else new_config_dict[key])

    return hps


def parser_add_adversarial_commons(parser):
    parser.add_argument('--acet_weight', type=float, default=1, help='Weight for out-distribution term in ACET (derivates)')
    parser.add_argument('--trades_weight', type=float, default=6, help='Weight for TRADES term in TRADES (derivates)')
    parser.add_argument('--train_clean', dest='train_clean', type=lambda x: bool(strtobool(x)),
                        default=False, help='Train on additional clean data or purely adversarial')
    parser.add_argument('--norm', type=str, default='l2',
                        help=('l2 or linf'))
    parser.add_argument('--od_bs_factor', default=1, type=float, help='OD batch size factor')

    #ID PGD for AT/TRADES
    parser.add_argument('--id_steps', type=int, default=20, help='steps in ID attack')
    parser.add_argument('--id_pgd', type=str, default='argmin',
                        help='PGD variation for ID attack: pgd, argmin, monotone')

    #OD PGD for ACET
    parser.add_argument('--od_steps', type=int, default=20, help='steps in OD attack (ACET)')
    parser.add_argument('--od_pgd', type=str, default='argmin',
                        help='PGD variation for OD attack: pgd, argmin, monotone')

    #Objectives
    parser.add_argument('--adv_obj', type=str, default='ce',
                        help=('Objective to optimize in the inner loop of adversarial training'
                              'logitsDiff | crossEntropy'))
    parser.add_argument('--acet_obj', type=str, default='KL',
                        help=('only for ACET; what objective the adversary has'
                              'conf | log_conf | entropy | KL | bhattacharyya'))

    parser.add_argument('--linf_eps', type=float, default=1.0, help='Linf epsilon')
    parser.add_argument('--l2_eps', type=float, default=1.0, help='L2 epsilon')
    parser.add_argument('--l1_eps', type=float, default=1.0, help='L1 epsilon')

    parser.add_argument('--linf_stepsize', type=float, default=1.0, help='Linf pgd stepsize')
    parser.add_argument('--l2_stepsize', type=float, default=1.0, help='L2 pgd stepsize')
    parser.add_argument('--l1_stepsize', type=float, default=1.0, help='L1 pgd stepsize')

    parser.add_argument('--od_eps_factor', type=float, default=1.0, help='Multiplier for ACET epsilon')



def load_model_checkpoint(model, model_dir, device, hps):
    # load old density_model
    if hps.continue_trained is not None:
        load_folder = hps.continue_trained[0]
        load_epoch = hps.continue_trained[1]
        start_epoch = int(int(hps.continue_trained[2]))  # * epoch_subdivs)
        if load_epoch in ['final', 'best', 'final_swa', 'best_swa']:
            state_dict_file = f'{model_dir}/{load_folder}/{load_epoch}.pth'
            optimizer_dict_file = f'{model_dir}/{load_folder}/{load_epoch}_optim.pth'
        else:
            state_dict_file = f'{model_dir}/{load_folder}/checkpoints/{load_epoch}.pth'
            optimizer_dict_file = f'{model_dir}/{load_folder}/checkpoints/{load_epoch}_optim.pth'

        state_dict = torch.load(state_dict_file, map_location=device)

        #if hps.continue_optim:
        try:
            optim_state_dict = torch.load(optimizer_dict_file, map_location=device)
        except:
            print('Warning: Could not load Optim State - Restarting optim')
            optim_state_dict = None
        #else:
        #    optim_state_dict = None

        model.load_state_dict(state_dict)

        print(f'Continuing {load_folder} from epoch {load_epoch} - Starting training at epoch {start_epoch}')
    else:
        start_epoch = 0
        optim_state_dict = None

    return start_epoch, optim_state_dict

def create_msda_config(hps):
    if hps.msda == 'none':
        msda_config = None
    elif hps.msda == 'fmix':
        if '_cutout' in hps.augm:
            augm_new = hps.augm.split('_cutout')[0]
            print(f'Warning found augmentation {hps.augm} with cutout - changing to {augm_new}')
            hps.augm = augm_new
        msda_config = msda.create_fmix_config()
    else:
        raise NotImplementedError()

    return msda_config

def create_optim_scheduler_swa_configs(hps):
    # SCHEDULER
    epochs = hps.epochs
    warmup_epochs = hps.warmup_epochs
    test_epochs = hps.test_epochs

    if hps.scheduler == 'step_lr':
        if epochs == 25:
            decay_epochs = [10, 15, 20]
            decay_rate = 0.1
        elif epochs == 100:
            decay_epochs = [50, 75, 90]
            decay_rate = 0.1
        elif epochs == 120:
            decay_epochs = [50, 85, 105]
            decay_rate = 0.1
        elif epochs == 150:
            decay_epochs = [60, 90, 120]
            decay_rate = 0.1
        elif epochs == 200:
            decay_epochs = [75, 125, 175]
            decay_rate = 0.1
        elif epochs == 220 or epochs == 250 or epochs == 230:
            decay_epochs = [100, 150, 200]
            decay_rate = 0.1
        elif epochs == 300:
            decay_epochs = [80, 160, 240]
            decay_rate = 0.2
        elif epochs == 320 or epochs == 350:
            decay_epochs = [150, 225, 300]
            decay_rate = 0.1
        elif epochs == 500:
            decay_epochs = [200, 300, 400]
            decay_rate = 0.1
        else:
            raise NotImplementedError()

        scheduler_config = schedulers.create_piecewise_consant_scheduler_config(epochs, decay_epochs, decay_rate,
                                                                                warmup_length=warmup_epochs)
    elif hps.scheduler == 'cosine':
        scheduler_config = schedulers.create_cosine_annealing_scheduler_config(epochs, 0.,
                                                                               warmup_length=warmup_epochs)
    else:
        raise NotImplementedError()

    # optimizer
    optimizer_config = optimizers.create_optimizer_config('SGD', hps.lr, momentum=hps.momentum, weight_decay=hps.decay,
                                                          nesterov=hps.nesterov, mixed_precision=hps.mixed_precision)

    if hps.swa == 'const':
        optimizers.config_creators.add_constant_swa_to_optimizer_config(hps.swa_epochs,
                                                                        hps.swa_update_frequency,
                                                                        hps.swa_virtual_schedule_lr,
                                                                        optimizer_config)
    elif hps.swa == 'cosine':
        optimizers.config_creators.add_cosine_swa_to_optimizer_config(hps.swa_epochs, hps.swa_cycle_length,
                                                                      hps.swa_update_frequency,
                                                                      hps.swa_virtual_schedule_length,
                                                                      hps.swa_virtual_schedule_swa_end,
                                                                      hps.swa_virtual_schedule_lr,
                                                                      optimizer_config)

    return scheduler_config, optimizer_config


def create_trainer(hps, model, optimizer_config, scheduler_config, device, num_classes, model_dir, log_dir, msda_config=None):
    id_l2_eps = hps.l2_eps
    od_l2_eps = hps.l2_eps * hps.od_eps_factor

    id_linf_eps = hps.linf_eps
    od_linf_eps = hps.linf_eps * hps.od_eps_factor

    id_l1_eps = hps.l1_eps
    od_l1_eps = hps.l1_eps * hps.od_eps_factor
    if hps.class_weights is not None:
        weights = torch.cuda.FloatTensor(hps.class_weights, device=device)
    else:
        weights = None
    print('weights are', weights)
    if hps.train_type == 'plain':
        trainer = tt.PlainTraining(model, optimizer_config, hps.epochs, device, num_classes,
                                   lr_scheduler_config=scheduler_config,
                                   msda_config=msda_config,
                                   saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs,
                                   weights=weights, clean_criterion=hps.clean_criterion)

    elif hps.train_type == 'adversarial':
        # https://arxiv.org/pdf/1906.09453.pdf
        if hps.norm in ['l2', '2']:
            attack_config = tt.AdversarialTraining.create_id_attack_config(id_l2_eps, hps.id_steps, hps.l2_stepsize, 'l2',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True)
        elif hps.norm in ['l1', '1']:
            attack_config = tt.AdversarialTraining.create_id_attack_config(id_l1_eps, hps.id_steps, hps.l1_stepsize, 'l1',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True)
        else:
            attack_config = tt.AdversarialTraining.create_id_attack_config(id_linf_eps, hps.id_steps, hps.linf_stepsize, 'inf',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True,
                                                                           momentum=0.0)  # , noise=f'uniform_{inf_eps}')

        trainer = tt.AdversarialTraining(model, attack_config, optimizer_config, hps.epochs, device, num_classes,
                                         train_clean=hps.train_clean, attack_loss=hps.adv_obj,
                                         lr_scheduler_config=scheduler_config,
                                         saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs,
                                         weights=weights, clean_criterion=hps.clean_criterion)
    elif hps.train_type in ['trades', 'TRADES']:
        if hps.norm in ['l2', '2']:
            attack_config = tt.TRADESTraining.create_id_attack_config(id_l2_eps, hps.id_steps, hps.l2_stepsize, 'l2',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True)
        elif hps.norm in ['l1', '1']:
            attack_config = tt.TRADESTraining.create_id_attack_config(id_l1_eps, hps.id_steps, hps.l1_stepsize, 'l1',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True)
        else:
            attack_config = tt.AdversarialTraining.create_id_attack_config(id_linf_eps, hps.id_steps, hps.linf_stepsize, 'inf',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True,
                                                                           momentum=0.0)
        trainer = tt.TRADESTraining(model, attack_config, optimizer_config, hps.epochs, device, num_classes,
                                    lr_scheduler_config=scheduler_config, trades_weight=hps.trades_weight,
                                    saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs,
                                    weights=weights, clean_criterion=hps.clean_criterion)
    elif hps.train_type in ['AdvACET', 'ADVACET']:
        # https://arxiv.org/pdf/1906.09453.pdf
        if hps.norm in ['l2', '2']:
            id_attack_config = tt.AdversarialACET.create_id_attack_config(id_l2_eps, hps.id_steps, hps.l2_stepsize, 'l2',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True)
            od_attack_config = tt.AdversarialACET.create_od_attack_config(od_l2_eps, hps.od_steps, hps.l2_stepsize, 'l2',
                                                                           pgd=hps.od_pgd,
                                                                           normalize_gradient=True)
        elif hps.norm in ['l1', '1']:
            id_attack_config = tt.AdversarialACET.create_id_attack_config(id_l1_eps, hps.id_steps, hps.l1_stepsize, 'l1',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True)
            od_attack_config = tt.AdversarialACET.create_od_attack_config(od_l1_eps, hps.od_steps, hps.l1_stepsize, 'l1',
                                                                           pgd=hps.od_pgd,
                                                                           normalize_gradient=True)
        else:
            id_attack_config = tt.AdversarialACET.create_id_attack_config(id_linf_eps, hps.id_steps, hps.linf_stepsize, 'inf',
                                                                           pgd=hps.id_pgd,
                                                                           normalize_gradient=True,
                                                                           momentum=0.0)
            od_attack_config = tt.AdversarialACET.create_od_attack_config(od_linf_eps, hps.od_steps, hps.linf_stepsize, 'inf',
                                                                           pgd=hps.od_pgd,
                                                                           normalize_gradient=True,
                                                                           momentum=0.0)
        trainer = tt.AdversarialACET(model, id_attack_config, od_attack_config, optimizer_config, hps.epochs, device,
                                     num_classes,
                                     train_clean=hps.train_clean, attack_loss=hps.adv_obj, lr_scheduler_config=scheduler_config,
                                     train_obj=hps.acet_obj, attack_obj=hps.acet_obj, lam=hps.acet_weight,
                                     saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type in ['tradesceda', 'TRADESCEDA']:
        raise NotImplementedError()
        trainer = tt.TRADESCEDATraining(model, id_attack_config, od_attack_config, optimizer_config, hps.epochs, device,
                                        num_classes,
                                        lr_scheduler_config=scheduler_config, train_obj=hps.t_obj, id_trades_weight=6,
                                        od_trades_weight=10,
                                        saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type == 'CEDA':
        trainer = tt.CEDATraining(model, optimizer_config, hps.epochs, device, lr_scheduler_config=scheduler_config,
                                  msda_config=msda_config,
                                  train_obj=hps.acet_obj, lam=hps.acet_weight, saved_model_dir=model_dir,
                                  saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type == 'ACET':
        # L2 disance between cifar10 and mnist is about 14 on average
        if hps.norm in ['l2', '2']:
            od_attack_config = tt.AdversarialACET.create_od_attack_config(od_l2_eps, hps.od_steps, hps.l2_stepsize, 'l2',
                                                                           pgd=hps.od_pgd,
                                                                           normalize_gradient=True)
        elif hps.norm in ['l1']:
            od_attack_config = tt.AdversarialACET.create_od_attack_config(od_l1_eps, hps.od_steps, hps.l1_stepsize, 'l1',
                                                                           pgd=hps.od_pgd,
                                                                           normalize_gradient=True)
        else:
            od_attack_config = tt.ACETTraining.create_od_attack_config(od_linf_eps, hps.od_steps, hps.linf_stepsize, 'inf',
                                                                           pgd=hps.od_pgd,
                                                                           normalize_gradient=True,
                                                                           momentum=0.0)
        trainer = tt.ACETTraining(model, od_attack_config, optimizer_config, hps.epochs, device, num_classes,
                                  lam=hps.acet_weight,
                                  lr_scheduler_config=scheduler_config, train_obj=hps.acet_obj, attack_obj=hps.acet_obj,
                                  saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    else:
        raise ValueError('Train type {} is not supported'.format(hps.train_type))

    return trainer
