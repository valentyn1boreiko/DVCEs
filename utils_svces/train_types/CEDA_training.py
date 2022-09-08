import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils_svces.distances as d

from .train_loss import MinMaxLoss, acet_uniform_obj_from_name, acet_target_obj_from_name, TrainLoss, entropy

from .out_distribution_training import OutDistributionTraining

class CEDAObjective(MinMaxLoss):
    def __init__(self, obj_str, K, log_stats=False, name_prefix=None):
        f, expected_format = acet_uniform_obj_from_name(obj_str, K)
        self.f = f
        super().__init__('CEDALoss_{}'.format(obj_str), expected_format=expected_format, log_stats=log_stats,
                         name_prefix=name_prefix)

    def inner_max(self, data, target):
        return data

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        obj_expanded =  self.f(self._prepare_input(model_out))
        self._log_stats(obj_expanded)
        return TrainLoss.reduce(obj_expanded, reduction)

class CEDATargetedObjective(MinMaxLoss):
    def __init__(self, obj_str, K, label_smoothing_eps=None, log_stats=False, name_prefix=None):
        #if targeted loss is false: enforce unfirom, otherwise divergence between y and softmax(model_out)
        f, expected_format = acet_target_obj_from_name(obj_str)
        self.K = K
        self.f = f
        self.label_smoothing_eps = label_smoothing_eps
        super().__init__('CEDALoss_{}'.format(obj_str), expected_format=expected_format, log_stats=log_stats,
                         name_prefix=name_prefix)

    def inner_max(self, data, target):
        return data

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        if y.shape[1] == self.K:
            weight = 1.0
            target = y
        else:
            weight = y[:, -1]
            target = y[:, 0:-1]

        if self.label_smoothing_eps is not None:
            target = (1-self.label_smoothing_eps) * target + self.label_smoothing_eps * 1/self.K

        obj_expanded =  weight * self.f(self._prepare_input(model_out), target)
        self._log_stats(obj_expanded)
        return TrainLoss.reduce(obj_expanded, reduction)


class CEDATargetedEntropyObjective(MinMaxLoss):
    def __init__(self, obj_str, K, entropy_weight=1.0, log_stats=False, name_prefix=None):
        f, expected_format = acet_target_obj_from_name(obj_str)
        self.K = K
        self.f = f
        self.entropy_weight = entropy_weight
        super().__init__('CEDALoss_{}'.format(obj_str), expected_format=expected_format, log_stats=log_stats,
                         num_losses=2, sub_losses_postfix=['', 'Entropy'], name_prefix=name_prefix)

    def inner_max(self, data, target):
        return data

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        if y.shape[1] == self.K:
            weight = 1.0
            target = y
        else:
            weight = y[:, -1]
            target = y[:, 0:-1]

        prep_out = self._prepare_input(model_out)
        obj_expanded = self.f(prep_out, target)
        entropy_expanded = entropy(model_out)
        loss_expanded = weight * (obj_expanded - self.entropy_weight * entropy_expanded)
        self._log_stats(obj_expanded, loss_idx=0)
        self._log_stats(entropy_expanded, loss_idx=1)
        return TrainLoss.reduce(loss_expanded, reduction)


class CEDATraining(OutDistributionTraining):
    def __init__(self, model, optimizer_config, epochs, device, num_classes,
                 CEDA_variant=None, lr_scheduler_config=None, msda_config=None, model_config=None,
                 clean_criterion='ce', train_obj='log_conf', od_weight=1., test_epochs=1, verbose=100,
                 saved_model_dir='SavedModels', saved_log_dir='Logs'):

        distance = d.LPDistance(p=2)
        super().__init__('CEDA', model, distance, optimizer_config, epochs, device, num_classes,
                         clean_criterion=clean_criterion, lr_scheduler_config=lr_scheduler_config,
                         msda_config=msda_config, model_config=model_config, od_weight=od_weight,
                         test_epochs=test_epochs, verbose=verbose, saved_model_dir=saved_model_dir,
                         saved_log_dir=saved_log_dir)

        if CEDA_variant is None:
            self.CEDA_variant = {'Type': 'CEDA'}
        else:
            self.CEDA_variant = CEDA_variant
        self.od_train_obj = train_obj


    def _get_od_criterion(self, epoch, model, name_prefix='OD'):
        if self.CEDA_variant['Type'] == 'CEDATargeted':
            label_smoothing_eps = None if 'LabelSmoothingEps' not in self.CEDA_variant.keys() else self.CEDA_variant['LabelSmoothingEps']
            train_criterion = CEDATargetedObjective(self.od_train_obj, self.classes,
                                                    label_smoothing_eps=label_smoothing_eps, log_stats=True,
                                                    name_prefix=name_prefix)
        elif self.CEDA_variant['Type'] == 'CEDA':
            train_criterion = CEDAObjective(self.od_train_obj, self.classes, log_stats=True, name_prefix='OD')
        elif self.CEDA_variant['Type'] == 'CEDATargetedEntropy':
            train_criterion = CEDATargetedEntropyObjective(self.od_train_obj, self.classes, log_stats=True,
                                                           name_prefix=name_prefix)
        else:
            raise NotImplementedError()

        return train_criterion

    def _get_CEDA_config(self):
        CEDA_config = {'CEDA Variant': self.CEDA_variant, 'train_obj': self.od_train_obj, 'lambda': self.od_weight}
        return CEDA_config

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        ceda_config = self._get_CEDA_config()

        configs = {}
        configs['Base'] = base_config
        configs['CEDA'] = ceda_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs





