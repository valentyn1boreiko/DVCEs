import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .in_distribution_training import InDistributionTraining
from .train_loss import MinMaxLoss, TrainLoss
from .helpers import get_distance


class RandomizedSmoothingLoss(MinMaxLoss):
    def __init__(self, noise_scales, log_stats=False, name_prefix=None):
        super().__init__('RandomizedSmoothingLoss', 'log_probabilities', log_stats=log_stats, name_prefix=name_prefix)
        self.noise_scales = torch.FloatTensor(noise_scales)

    def inner_max(self, data, target):
        chosen_noise_scale = torch.randint(len(self.noise_scales), (data.shape[0],))
        noise_eps = self.noise_scales[chosen_noise_scale].view(data.shape[0], *([1] * len(data.shape[1:]))).to(data.device)
        adv_samples = (data + noise_eps * torch.randn_like(data)).clamp(0.0, 1.0)
        return adv_samples

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = F.cross_entropy(prep_out, y, reduction='none' )
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class RandomizedSmoothingTraining(InDistributionTraining):
    def __init__(self, model, optimizer_config, epochs, device, num_classes, noise_scales, train_clean=True,
                 lr_scheduler_config=None, model_config=None,
                 test_epochs=1, verbose=100, saved_model_dir='SavedModels',
                 saved_log_dir='Logs'):

        distance = get_distance('l2')
        self.noise_scales = noise_scales

        super().__init__('RandomizedSmoothing', model, distance, optimizer_config, epochs, device, num_classes,
                         train_clean=train_clean, lr_scheduler_config=lr_scheduler_config, model_config=model_config,
                         test_epochs=test_epochs, verbose=verbose, saved_model_dir=saved_model_dir,
                         saved_log_dir=saved_log_dir)


    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
        id_train_criterion = RandomizedSmoothingLoss(self.noise_scales, log_stats=True, name_prefix=name_prefix)
        return id_train_criterion

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        adv_config = self._get_randomized_smoothing_training_config()

        configs = {}
        configs['Base'] = base_config
        configs['Randomized Smoothing Training'] = adv_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config
        configs['MSDA'] = self.msda_config

        configs['Data Loader'] = loader_config
        configs['Model'] = self.model_config

        return configs

    def _get_randomized_smoothing_training_config(self):
        config_dict = {}
        config_dict['Num Noise scales'] = len(self.noise_scales)
        config_dict['Min Noise scales'] = torch.min(self.noise_scales).item()
        config_dict['Max Noise scales'] = torch.max(self.noise_scales).item()
        return config_dict




