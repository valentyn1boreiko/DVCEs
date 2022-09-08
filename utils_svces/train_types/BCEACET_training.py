import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils_svces.distances as d

from .out_distribution_training import OutDistributionTraining
from .train_loss import TrainLoss, MinMaxLoss, NegativeWrapper, BCELogitsProxy
from .helpers import create_attack_config, get_adversarial_attack, get_distance


class BCEACETObjective(MinMaxLoss):
    def __init__(self,  model, epoch, attack_config, mask_features, min_features_mask, max_features_mask, num_features,
                 log_stats=False, number_of_batches=None, name_prefix=None):
        super().__init__('BCEACET', expected_format='logits', log_stats=log_stats, name_prefix=name_prefix)
        self.attack_config = attack_config
        self.model = model
        self.epoch = epoch
        self.num_features = num_features
        self.mask_features = mask_features
        self.min_features_mask = min_features_mask
        self.max_features_mask = max_features_mask


    def _generate_mask(self, target):
        if self.mask_features:
            mask = torch.zeros_like(target, dtype=torch.float)
            for idx in range(mask.shape[0]):
                num_non_masked = torch.randint(self.min_features_mask, self.max_features_mask + 1, (1,)).item()
                non_masked_idcs = torch.randperm(mask.shape[1], device=mask.device)[:num_non_masked]
                mask[idx, non_masked_idcs] = 1

            self.mask = mask
        else:
            self.mask = None

    def inner_max(self, data, target):
        uniform_target = 0.5 * data.new_ones((data.shape[0],self.num_features), dtype=torch.float)
        self._generate_mask(uniform_target)
        self.obj = BCELogitsProxy(mask=self.mask, log_stats=False)
        neg_obj =  NegativeWrapper(self.obj)
        self.attack = get_adversarial_attack(self.attack_config, self.model, neg_obj,
                                             num_classes=self.num_classes, epoch=self.epoch)
        adv_samples = self.attack(data, uniform_target, targeted=False)
        return adv_samples


    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        assert self.obj is not None, 'Inner Max has to be called first'
        uniform_target = 0.5 * data.new_ones((data.shape[0],self.num_features), dtype=torch.float)
        loss_expanded = self.obj(None, prep_out, None, uniform_target, reduction='none' )
        self.obj = None
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)


class BCEACETTraining(OutDistributionTraining):
    def __init__(self, model, od_attack_config, optimizer_config, epochs, device, num_classes,
                 mask_features, min_features_mask, max_features_mask, lr_scheduler_config=None, lam=1.,
                 test_epochs=5, verbose=100,
                 saved_model_dir='SavedModels', saved_log_dir='Logs'):

        distance = get_distance(od_attack_config['norm'])

        super().__init__('BCEACET', model, distance, optimizer_config, epochs, device, num_classes,
                         clean_criterion='bce', lr_scheduler_config=lr_scheduler_config, od_weight=lam,
                         test_epochs=test_epochs, verbose=verbose, saved_model_dir=saved_model_dir,
                         saved_log_dir=saved_log_dir)

        self.od_attack_config = od_attack_config
        self.mask_featrues = mask_features
        self.min_features_mask = min_features_mask
        self.max_features_mask = max_features_mask


    def _get_od_criterion(self, epoch, model):
        train_criterion = BCEACETObjective(model, epoch, self.od_attack_config, self.mask_featrues,
                                           self.min_features_mask, self.max_features_mask, self.classes,
                                           log_stats=True, name_prefix='OD')
        return train_criterion

    def _get_od_attack(self, epoch, att_criterion):
        return get_adversarial_attack(self.od_attack_config, self.model, att_criterion, num_classes=self.classes, epoch=epoch)

    def _get_ACET_config(self):
        ACET_config = {'lambda': self.od_weight}
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
        configs['Model'] = model_config

        return configs