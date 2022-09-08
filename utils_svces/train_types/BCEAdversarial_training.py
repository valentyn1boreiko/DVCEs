import torch

from .helpers import get_adversarial_attack, create_attack_config, get_distance
from .in_distribution_training import InDistributionTraining
from .train_loss import TrainLoss, MinMaxLoss, BCELogitsProxy, NegativeWrapper, BCAccuracyConfidenceLogger


class BCEAdversarialLoss(MinMaxLoss):
    def __init__(self, model, epoch, attack_config, mask_features, min_features_mask, max_features_mask, num_classes,
                 log_stats=False,name_prefix=None):
        super().__init__('AdversarialLoss', 'logits', log_stats=log_stats, name_prefix=name_prefix)
        self.attack_config = attack_config
        self.model = model
        self.epoch = epoch
        self.mask_features = mask_features
        self.min_features_mask = min_features_mask
        self.max_features_mask = max_features_mask
        self.num_classes = num_classes
        self.obj = None

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
        self._generate_mask(target)
        self.obj = BCELogitsProxy(mask=self.mask, log_stats=False)
        neg_obj = NegativeWrapper(self.obj)
        self.attack = get_adversarial_attack(self.attack_config, self.model, neg_obj,
                                             num_classes=self.num_classes, epoch=self.epoch)
        adv_samples = self.attack(data, target, targeted=False)

        return adv_samples

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        assert self.obj is not None, 'Inner Max has to be called first'
        loss_expanded = self.obj(data, prep_out, orig_data, y, reduction='none')
        self.obj = None
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)


class BCEAdversarialTraining(InDistributionTraining):
    def __init__(self, model, id_attack_config, optimizer_config, epochs, device, num_classes, train_clean=True,
                  lr_scheduler_config=None, model_config=None,
                 test_epochs=1, verbose=100, saved_model_dir='SavedModels',
                 saved_log_dir='Logs'):
        distance = get_distance(id_attack_config['norm'])

        super().__init__('BCEAdversarial Training', model, distance, optimizer_config,
                         epochs, device, num_classes,
                         clean_criterion='bce', train_clean=train_clean,
                         lr_scheduler_config=lr_scheduler_config, model_config=model_config,
                         test_epochs=test_epochs, verbose=verbose, saved_model_dir=saved_model_dir,
                         saved_log_dir=saved_log_dir)
        self.id_attack_config = id_attack_config

    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
        id_train_criterion = BCEAdversarialLoss(model, epoch, self.id_attack_config, True, 1, 1, self.classes,
                                                log_stats=True, name_prefix=name_prefix)
        return id_train_criterion

    def _get_id_accuracy_conf_logger(self, name_prefix):
        return BCAccuracyConfidenceLogger(self.classes, name_prefix=name_prefix)

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        adv_config = self._get_adversarial_training_config()

        configs = {}
        configs['Base'] = base_config
        configs['Adversarial Training'] = adv_config
        configs['ID Attack'] = self.id_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs

    def _get_adversarial_training_config(self):
        config_dict = {}
        config_dict['train_clean'] = self.train_clean
        return config_dict
