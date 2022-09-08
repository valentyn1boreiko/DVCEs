import torch
from .in_distribution_training import InDistributionTraining
from .helpers import get_distance
from .train_loss import ZeroLoss

class PlainTraining(InDistributionTraining):
    def __init__(self, model, optimizer_config, epochs, device, num_classes, clean_criterion='ce',
                 lr_scheduler_config=None, msda_config=None, model_config=None, test_epochs=1, verbose=100, saved_model_dir='SavedModels',
                 saved_log_dir='Logs'):
        distance = get_distance('l2')
        super().__init__('plain', model, distance, optimizer_config, epochs, device, num_classes,
                         train_clean=False, id_trades=True, clean_weight=1., id_adv_weight=0.,
                         lr_scheduler_config=lr_scheduler_config, msda_config=msda_config, model_config=model_config,
                         test_epochs=test_epochs, clean_criterion=clean_criterion,
                         verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
        id_train_criterion = ZeroLoss()
        return id_train_criterion

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        plain_config = self._get_plain_config()

        configs = {}
        configs['Base'] = base_config
        configs['Plain Training'] = plain_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config
        configs['MSDA'] = self.msda_config
        configs['Data Loader'] = loader_config
        configs['Model'] = self.model_config


        return configs

    def _get_plain_config(self):
        config_dict = {}
        config_dict['Clean Weight'] = self.clean_weight
        return config_dict




