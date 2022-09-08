import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .in_distribution_training import InDistributionTraining
from .train_loss import MinMaxLoss, TrainLoss
from .helpers import get_adversarial_attack, create_attack_config, get_distance

from utils_svces.adversarial_attacks import *
from utils_svces.distances import LPDistance

class AdversarialLoss(MinMaxLoss):
    def __init__(self, model, epoch, attack_config, num_classes, inner_objective='crossentropy', log_stats=False, name_prefix=None):
        super().__init__('AdversarialLoss', 'log_probabilities', log_stats=log_stats, name_prefix=name_prefix)
        self.attack = get_adversarial_attack(attack_config, model, inner_objective, num_classes=num_classes, epoch=epoch)

    def inner_max(self, data, target):
        adv_samples = self.attack(data, target, targeted=False)
        return adv_samples

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = F.cross_entropy(prep_out, y, reduction='none' )
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class AdversarialTraining(InDistributionTraining):
    def __init__(self, model, id_attack_config, optimizer_config, epochs, device, num_classes, train_clean=True,
                attack_loss='logits_diff', lr_scheduler_config=None, model_config=None,
                 test_epochs=1, verbose=100, saved_model_dir='SavedModels',
                 saved_log_dir='Logs'):

        distance = get_distance(id_attack_config['norm'])

        if train_clean:
            clean_weight = 0.5
            adv_weight = 0.5
        else:
            clean_weight = 0.0
            adv_weight = 1.0

        super().__init__('Adversarial Training', model, distance, optimizer_config, epochs, device, num_classes,
                         train_clean=train_clean, clean_weight=clean_weight, id_adv_weight=adv_weight,
                         lr_scheduler_config=lr_scheduler_config, model_config=model_config,
                         test_epochs=test_epochs, verbose=verbose, saved_model_dir=saved_model_dir,
                         saved_log_dir=saved_log_dir)
        self.id_attack_config = id_attack_config
        self.attack_loss = attack_loss


    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
        id_train_criterion = AdversarialLoss(model, epoch, self.id_attack_config, self.classes, inner_objective=self.attack_loss,
                                             log_stats=True, name_prefix=name_prefix)
        return id_train_criterion

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        adv_config = self._get_adversarial_training_config()

        configs = {}
        configs['Base'] = base_config
        configs['Adversarial Training'] = adv_config
        configs['ID Attack'] = self.id_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config
        configs['MSDA'] = self.msda_config
        configs['Data Loader'] = loader_config
        configs['Model'] = self.model_config


        return configs

    def _get_adversarial_training_config(self):
        config_dict = {}
        config_dict['Train Clean'] = self.train_clean
        config_dict['Adversarial Loss'] = self.attack_loss
        config_dict['Clean Weight'] = self.clean_weight
        config_dict['Adversarial Weight'] = self.id_adv_weight
        return config_dict




