import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils_svces.distances as d

from .ACET_training import ACETObjective, ACETTargetedObjective
from .Adversarial_training import AdversarialTraining, AdversarialLoss
from .in_out_distribution_training import InOutDistributionTraining
from .helpers import get_adversarial_attack, create_attack_config, get_distance
from .train_loss import CrossEntropyProxy


######################################################
class AdversarialACET(InOutDistributionTraining):
    def __init__(self, model, id_attack_config, od_attack_config, optimizer_config, epochs, device, num_classes,
                 train_clean=True,
                 attack_loss='LogitsDiff', lr_scheduler_config=None, model_config=None,
                 target_confidences=False,
                 attack_obj='log_conf', train_obj='log_conf', od_weight=1., test_epochs=1, verbose=100,
                 saved_model_dir='SavedModels', saved_log_dir='Logs'):

        id_distance = get_distance(id_attack_config['norm'])
        od_distance = get_distance(od_attack_config['norm'])

        if train_clean:
            id_clean_weight = 1.0
            id_adv_weight = 1.0
        else:
            id_clean_weight = 0.0
            id_adv_weight = 1.0

        super().__init__('AdvACET', model, id_distance, od_distance, optimizer_config, epochs, device, num_classes,
                         train_clean=train_clean, id_weight=0.5, id_adv_weight=id_adv_weight, clean_weight=id_clean_weight,
                         od_weight=0.5 * od_weight, od_clean_weight=0.0, od_adv_weight=1.0,
                         lr_scheduler_config=lr_scheduler_config,
                         model_config=model_config, test_epochs=test_epochs, verbose=verbose,
                         saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

        # Adversarial specific
        self.id_attack_config = id_attack_config
        self.attack_loss = attack_loss

        # ACET specifics
        self.target_confidences = target_confidences
        self.od_attack_config = od_attack_config
        self.od_attack_obj = attack_obj
        self.od_train_obj = train_obj


    def _get_adversarialacet_config(self):
        config_dict = {}
        config_dict['Train Clean'] = self.train_clean
        config_dict['Adversarial Loss'] = self.attack_loss
        config_dict['ID Weight'] = self.id_weight
        config_dict['Clean Weight'] = self.clean_weight
        config_dict['Adversarial Weight'] = self.id_adv_weight

        config_dict['OD Targeted Confidences'] = self.target_confidences
        config_dict['OD Train Objective'] = self.od_train_obj
        config_dict['OD Attack_obj'] = self.od_attack_obj
        config_dict['OD Weight'] = self.od_weight
        config_dict['OD Adversarial Weight'] = self.od_adv_weight

        return config_dict

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()

        configs = {}
        configs['Base'] = base_config
        configs['ID Attack'] = self.id_attack_config
        configs['AdversarialACET'] = self._get_adversarialacet_config()
        configs['OD Attack'] = self.od_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs

    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
        id_train_criterion = AdversarialLoss(model, epoch, self.id_attack_config, self.classes,
                                             inner_objective=self.attack_loss,
                                             log_stats=True, name_prefix=name_prefix)
        return id_train_criterion

    def _get_od_clean_criterion(self, epoch, model, name_prefix='OD'):
        return None

    def _get_od_criterion(self, epoch, model, name_prefix='OD'):
        if self.target_confidences:
            train_criterion = ACETTargetedObjective(model, epoch, self.od_attack_config, self.od_train_obj,
                                                    self.od_attack_obj, self.classes,
                                                    log_stats=True, name_prefix=name_prefix)
        else:
            train_criterion = ACETObjective(model, epoch, self.od_attack_config, self.od_train_obj,
                                            self.od_attack_obj, self.classes,
                                            log_stats=True, name_prefix=name_prefix)
        return train_criterion

    def _get_od_attack(self, epoch, att_criterion):
        return get_adversarial_attack(self.od_attack_config, self.model, att_criterion, num_classes=self.classes,
                                      epoch=epoch)
