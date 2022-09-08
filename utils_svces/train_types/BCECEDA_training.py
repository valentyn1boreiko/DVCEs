import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils_svces.distances as d

from .out_distribution_training import OutDistributionTraining
from .train_loss import TrainLoss, MinMaxLoss, BCELogitsProxy

class BCEACETObjective(MinMaxLoss):
    def __init__(self,  epoch, mask_features, min_features_mask, max_features_mask, num_features,
                 log_stats=False, number_of_batches=None, name_prefix=None):
        super().__init__('BCEACET', expected_format='logits', log_stats=log_stats, name_prefix=name_prefix)
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
        return data

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        uniform_target = 0.5 * data.new_ones((data.shape[0],self.num_features), dtype=torch.float)
        self._generate_mask(uniform_target)
        obj = BCELogitsProxy(mask=self.mask, log_stats=False)
        loss_expanded = obj(None, prep_out, None, uniform_target, reduction='none' )
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class BCECEDATraining(OutDistributionTraining):
    def __init__(self, model, optimizer_config, epochs, device, num_classes,
                 mask_features, min_features_mask, max_features_mask,
                 lr_scheduler_config=None,
                 lam=1., test_epochs=1, verbose=100, saved_model_dir= 'SavedModels', saved_log_dir= 'Logs'):

        distance = d.LPDistance(p=2)
        super().__init__('BCECEDA', model, distance, optimizer_config, epochs, device, num_classes,
                         clean_criterion='bce', lr_scheduler_config=lr_scheduler_config, od_weight=lam,
                         test_epochs=test_epochs, verbose=verbose, saved_model_dir=saved_model_dir,
                         saved_log_dir=saved_log_dir)

        self.mask_features = mask_features
        self.min_features_mask = min_features_mask
        self.max_features_mask = max_features_mask

    def _get_od_criterion(self, epoch, model):
        train_criterion = BCEACETObjective(epoch, self.mask_features, self.min_features_mask, self.max_features_mask, self.classes,
                                           log_stats=True, name_prefix='OD')
        return None, train_criterion

    def _get_CEDA_config(self):
        CEDA_config = {'lambda': self.od_weight}
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
        configs['Model'] = model_config

        return configs





