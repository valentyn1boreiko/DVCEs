import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils_svces.distances as d

from .ACET_training import ACETObjective
from .CEDA_training import CEDAObjective
from .TRADES_training import TRADESLoss
from .in_out_distribution_training import InOutDistributionTraining
from .train_loss import AccuracyConfidenceLogger, DistanceLogger, ConfidenceLogger, SingleValueLogger, NegativeWrapper
from .helpers import interleave_forward, get_distance
import torch.cuda.amp as amp

import math
######################################################
class TRADESACETTraining(InOutDistributionTraining):
    def __init__(self, model, id_attack_config, od_attack_config, optimizer_config, epochs, device, num_classes,
                 trades_weight=1, lr_scheduler_config=None,
                 acet_obj='kl', od_weight=1., model_config=None,
                 test_epochs=1, verbose=100, saved_model_dir= 'SavedModels', saved_log_dir= 'Logs'):

        id_distance = get_distance(id_attack_config['norm'])
        od_distance = get_distance(od_attack_config['norm'])


        super().__init__('TRADESACET', model, id_distance, od_distance, optimizer_config, epochs, device, num_classes,
                         train_clean=False, id_trades=True, id_weight=0.5, clean_weight=1.0, id_adv_weight=trades_weight,
                         od_trades=False, od_weight=0.5 * od_weight, od_clean_weight=0.0, od_adv_weight=1.0,
                          lr_scheduler_config=lr_scheduler_config,
                          model_config=model_config, test_epochs=test_epochs,
                          verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

        #Trades
        self.id_attack_config = id_attack_config

        #od
        self.od_attack_config = od_attack_config
        self.acet_obj = acet_obj

    def requires_out_distribution(self):
        return True

    def create_loaders_dict(self, train_loader, test_loader=None, out_distribution_loader=None, out_distribution_test_loader=None, *args, **kwargs):
        train_loaders = {
            'train_loader': train_loader,
            'out_distribution_loader': out_distribution_loader
        }

        test_loaders = {}
        if test_loader is not None:
            test_loaders['test_loader'] = test_loader
        if out_distribution_test_loader is not None:
            test_loaders['out_distribution_test_loader'] = out_distribution_test_loader

        return train_loaders, test_loaders

    def _validate_loaders(self, train_loaders, test_loaders):
        if not 'train_loader' in train_loaders:
            raise ValueError('Train loader not given')
        if not 'out_distribution_loader' in train_loaders:
            raise ValueError('Out distribution loader is required for out distribution training')

    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
        trades_reg = TRADESLoss(model, epoch, self.id_attack_config, self.classes, log_stats=True, name_prefix=name_prefix)
        return trades_reg

    def _get_od_clean_criterion(self, epoch, model, name_prefix='OD'):
        od_clean_criterion = None
        return od_clean_criterion

    def _get_od_criterion(self, epoch, model,  name_prefix='OD'):
        od_criterion = ACETObjective(model, epoch, self.od_attack_config, self.acet_obj, self.acet_obj,
                                            self.classes, log_stats=True, name_prefix=name_prefix)
        return od_criterion

    def _get_TRADESACET_config(self):
        config_dict = {}
        config_dict['ID Weight'] = self.id_weight
        config_dict['Clean Weight'] = self.clean_weight
        config_dict['Trades Weight'] = self.id_adv_weight

        config_dict['OD Weight'] = self.od_weight
        config_dict['OD ACET Weight'] = self.od_adv_weight
        config_dict['OD ACET Objective'] = self.acet_obj
        return config_dict

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        tradesacet_config = self._get_TRADESACET_config()
        configs = {}
        configs['Base'] = base_config
        configs['TRADESACET'] = tradesacet_config
        configs['ID Attack'] = self.id_attack_config
        configs['OD Attack'] = self.od_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs

