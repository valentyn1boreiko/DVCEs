import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils_svces.distances as d

from .out_distribution_training import OutDistributionTraining
from .train_loss import acet_uniform_obj_from_name, acet_target_obj_from_name, MinMaxLoss, TrainLoss
from .train_loss import LoggingLoss, NegativeWrapper
from .helpers import create_attack_config, get_adversarial_attack, get_distance


################################################ACET##############################################
class ACETObjective(MinMaxLoss):
    def __init__(self, model, epoch, attack_config, train_obj, attack_obj, num_classes, log_stats=False, number_of_batches=None, name_prefix=None):
        f_train, train_expected_format = acet_uniform_obj_from_name(train_obj, num_classes)
        f_attack, attack_expected_format = acet_uniform_obj_from_name(attack_obj, num_classes)

        self.f_train = f_train
        super().__init__('ACETLoss', expected_format=train_expected_format, log_stats=log_stats,
                         name_prefix=name_prefix)

        #negative loss for inner maximization problem
        def att_criterion(data, model_out, orig_data, y, reduction='mean'):
            obj_expanded = -f_attack(self._prepare_input(model_out))
            return TrainLoss.reduce(obj_expanded, reduction)

        self.adv_attack = get_adversarial_attack(attack_config, model, att_criterion, num_classes=num_classes, epoch=epoch)


    def inner_max(self, data, target):
        adv_samples = self.adv_attack(data, target, targeted=False)
        return adv_samples

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        obj_expanded =  self.f_train(self._prepare_input(model_out))
        self._log_stats(obj_expanded)
        return TrainLoss.reduce(obj_expanded, reduction)

class ACETTargetedObjective(MinMaxLoss):
    def __init__(self, model, epoch, attack_config, train_obj, attack_obj, num_classes, log_stats=False, number_of_batches=None, name_prefix=None):
        #if targeted loss is false: enforce unfirom, otherwise divergence between y and softmax(model_out)
        f_train, train_expected_format = acet_target_obj_from_name(train_obj)
        f_attack, attack_expected_format = acet_target_obj_from_name(attack_obj)

        self.f_train = f_train
        super().__init__('ACETLoss', expected_format=train_expected_format, log_stats=log_stats,
                         name_prefix=name_prefix)

        # negative loss for inner maximization problem
        def att_criterion(data, model_out, orig_data, y, reduction='mean'):
            obj_expanded = -f_attack(self._prepare_input(model_out), y)
            return TrainLoss.reduce(obj_expanded, reduction)

        self.adv_attack = get_adversarial_attack(attack_config, model, att_criterion, num_classes=num_classes,
                                                 epoch=epoch)

    def inner_max(self, data, target):
        adv_samples = self.adv_attack(data, target, targeted=False)
        return adv_samples


    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        obj_expanded = self.f_train(self._prepare_input(model_out), y)
        self._log_stats(obj_expanded)
        return TrainLoss.reduce(obj_expanded, reduction)


class ACETTraining(OutDistributionTraining):
    def __init__(self, model, od_attack_config, optimizer_config, epochs, device, num_classes,
                 lr_scheduler_config=None, model_config=None, target_confidences=False, od_weight=1.,
                 train_obj='KL', attack_obj='KL',test_epochs=5, verbose=100,
                 saved_model_dir='SavedModels', saved_log_dir='Logs'):

        distance = get_distance(od_attack_config['norm'])

        super().__init__('ACET', model, distance, optimizer_config, epochs, device, num_classes,
                         lr_scheduler_config=lr_scheduler_config, model_config=model_config, od_weight=od_weight,
                         test_epochs=test_epochs, verbose=verbose, saved_model_dir=saved_model_dir,
                         saved_log_dir=saved_log_dir)

        self.od_attack_config = od_attack_config

        self.target_confidences = target_confidences
        self.od_attack_obj = attack_obj
        self.od_train_obj = train_obj

    def _get_od_criterion(self, epoch, model,  name_prefix='OD'):
        if self.target_confidences:
            train_criterion = ACETTargetedObjective(model, epoch, self.od_attack_config, self.od_train_obj, self.od_attack_obj, self.classes,
                                        log_stats=True, name_prefix=name_prefix)
        else:
            train_criterion = ACETObjective(model, epoch, self.od_attack_config, self.od_train_obj, self.od_attack_obj, self.classes,
                                        log_stats=True, name_prefix=name_prefix)
        return train_criterion

    def _get_ACET_config(self):
        ACET_config = {'targeted confidences': self.target_confidences, 'train_obj': self.od_train_obj, 'attack_obj': self.od_attack_obj, 'lambda': self.od_weight}
        return ACET_config

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        ACET_config = self._get_ACET_config()

        configs = {}
        configs['Base'] = base_config
        configs['ACET'] = ACET_config
        configs['OD Attack'] = self.od_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs