import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .in_distribution_training import InDistributionTraining
from .train_loss import LoggingLoss, TrainLoss, CrossEntropyProxy, AccuracyConfidenceLogger, DistanceLogger,\
    SingleValueLogger, KLDivergenceProxy, NegativeWrapper, MinMaxLoss
from .helpers import interleave_forward, get_adversarial_attack, create_attack_config, get_distance

class TRADESLoss(MinMaxLoss):
    def __init__(self,  model, epoch, attack_config, num_classes, log_stats=False, name_prefix=None):
        super().__init__('TRADES', 'logits', log_stats=log_stats, name_prefix=name_prefix)
        self.model = model
        self.epoch = epoch
        self.attack_config = attack_config

        self.div = KLDivergenceProxy(log_stats=False)
        self.adv_attack = get_adversarial_attack(self.attack_config, self.model, 'kl', num_classes, epoch=self.epoch)

    def inner_max(self, data, target):
        is_train = self.model.training
        #attack is run in test mode so target distribution should also be estimated in test not train
        self.model.eval()
        target_distribution = F.softmax(self.model(data), dim=1).detach()
        x_adv = self.adv_attack(data, target_distribution)

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        return x_adv.detach()

    #model out will be model out at adversarial samples
    #y will be the softmax distribution at original datapoint
    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = self.div(data, prep_out, orig_data, y, reduction='none')
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

#Base class for train types that use custom losses/attacks on the in distribution such as adversarial training
class TRADESTraining(InDistributionTraining):
    def __init__(self, model, id_attack_config, optimizer_config, epochs, device, num_classes, trades_weight=1.,
                 lr_scheduler_config=None, model_config=None, test_epochs=1, verbose=100,
                 saved_model_dir= 'SavedModels', saved_log_dir= 'Logs'):

        distance = get_distance(id_attack_config['norm'])

        super().__init__('TRADES', model, distance, optimizer_config, epochs, device, num_classes,
                         train_clean=False, id_trades=True, clean_weight=1.0, id_adv_weight=trades_weight,
                         lr_scheduler_config=lr_scheduler_config, model_config=model_config, test_epochs=test_epochs,
                         verbose=verbose, saved_model_dir= saved_model_dir, saved_log_dir=saved_log_dir)

        self.id_attack_config = id_attack_config

    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
        trades_reg = TRADESLoss(model, epoch, self.id_attack_config, self.classes, log_stats=True, name_prefix=name_prefix)
        return trades_reg

    def _get_TRADES_config(self):
        config_dict = {}
        config_dict['Clean Weight'] = self.clean_weight
        config_dict['Trades Weight']  = self.id_adv_weight
        return config_dict

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        adv_config = self._get_TRADES_config()

        configs = {}
        configs['Base'] = base_config
        configs['TRADES'] = adv_config
        configs['ID Attack'] = self.id_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs
