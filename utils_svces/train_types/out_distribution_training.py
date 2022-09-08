import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp

from torch.utils.tensorboard import SummaryWriter

import time

from .train_type import TrainType
from .train_loss import ConfidenceLogger, BCConfidenceLogger, DistanceLogger, SingleValueLogger
from .helpers import interleave_forward


###################################OUT DISTRIBUTION TRAINING#########################################
class OutDistributionTraining(TrainType):
    def __init__(self, name, model, od_distance, optimizer_config, epochs, device, num_classes, clean_criterion='ce',
                 lr_scheduler_config=None, msda_config=None, model_config=None, id_weight=1., od_weight=1.,
                 test_epochs=5, verbose=100, saved_model_dir='SavedModels', saved_log_dir='Logs'):

        super().__init__(name, model, optimizer_config, epochs, device, num_classes,
                         clean_criterion=clean_criterion, lr_scheduler_config=lr_scheduler_config,
                         msda_config=msda_config, model_config=model_config, test_epochs=test_epochs,
                         verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

        self.id_weight = id_weight
        self.od_weight = od_weight
        self.od_distance = od_distance
        self.od_iterator = None

    def requires_out_distribution(self):
        return True

    def _get_od_criterion(self, epoch, model,  name_prefix='OD'):
        #Should return a MinMaxLoss
        raise NotImplementedError()

    def _get_od_conf_logger(self, name_prefix=None):
        if self.clean_criterion in ['ce', 'crossentropy', 'kl', 'KL'] or 'klEntropy' in self.clean_criterion:
            return ConfidenceLogger(name_prefix=name_prefix)
        elif self.clean_criterion == 'bce':
            return BCConfidenceLogger(self.classes, name_prefix=name_prefix)
        else:
            raise NotImplementedError()

    def test(self, test_loaders, epoch, test_avg_model=False):
        # test set accuracy
        new_best = super().test(test_loaders, epoch, test_avg_model)

        if test_avg_model:
            model = self.avg_model
        else:
            model = self.model

        model.eval()

        if 'out_distribution_test_loader' in test_loaders:
            out_distribution_test_loader = test_loaders['out_distribution_test_loader']
            # other accuracy
            if test_avg_model:
                prefix = 'AVG_OD'
            else:
                prefix = 'OD'

            od_train_criterion = self._get_od_criterion(epoch, model)
            losses = [od_train_criterion]

            distance_od = DistanceLogger(self.od_distance, name_prefix=prefix)
            confidence_od = self._get_od_conf_logger(name_prefix=prefix)
            loggers = [distance_od, confidence_od]


            test_set_batches = len(out_distribution_test_loader)
            self.output_backend.start_epoch_log(test_set_batches)

            with torch.no_grad():
                for batch_idx, (od_data, od_target) in enumerate(out_distribution_test_loader):
                    od_data, od_target = od_data.to(self.device), od_target.to(self.device)

                    adv_noise = od_train_criterion.inner_max(od_data, od_target)
                    out = model(adv_noise)

                    od_train_criterion(adv_noise, out, od_data, od_target)

                    distance_od(adv_noise, out, od_data, od_target)
                    confidence_od(adv_noise, out, od_data, od_target)
                    self.output_backend.log_batch_summary(epoch, batch_idx, False, losses=losses, loggers=loggers)

            self.output_backend.end_epoch_write_summary(losses, loggers, epoch, False)

        return new_best

    def create_loaders_dict(self, train_loader, test_loader=None, out_distribution_loader=None,
                            out_distribution_test_loader=None, extra_test_loaders=None,*args, **kwargs):
        train_loaders = {
            'train_loader': train_loader,
            'out_distribution_loader': out_distribution_loader
        }

        test_loaders = {}
        if test_loader is not None:
            test_loaders['test_loader'] = test_loader
        if out_distribution_test_loader is not None:
            test_loaders['out_distribution_test_loader'] = out_distribution_test_loader
        if extra_test_loaders is not None:
            test_loaders['extra_test_loaders'] = extra_test_loaders

        return train_loaders, test_loaders

    def _validate_loaders(self, train_loaders, test_loaders):
        if not 'train_loader' in train_loaders:
            raise ValueError('Train cifar_loader not given')
        if not 'out_distribution_loader' in train_loaders:
            raise ValueError('Out distribution cifar_loader is required for out distribution training')

    def __get_loss_closure(self, clean_data, clean_target,
                           od_adv_samples, od_data, od_target,
                           clean_loss, od_train_criterion,
                           total_loss_logger=None,
                           lr_logger=None,
                           acc_conf_clean=None,
                           confidence_od=None,
                           distance_od=None):

        def loss_closure(log=False):
            clean_out, od_out = interleave_forward(self.model, [clean_data, od_adv_samples], in_parallel=self.in_parallel)

            loss1 = clean_loss(clean_data, clean_out, clean_data, clean_target)
            loss2 = od_train_criterion(od_adv_samples, od_out, od_data, od_target)

            loss = self.id_weight * loss1 + self.od_weight * loss2

            if log:
                # log
                total_loss_logger.log(loss)
                lr_logger.log(self.scheduler.get_last_lr()[0])

                acc_conf_clean(clean_data, clean_out, clean_data, clean_target)
                confidence_od(od_adv_samples, od_out, od_data, od_target)
                distance_od(od_adv_samples, od_out, od_data, od_target)

            return loss

        return loss_closure

    def _inner_train(self, train_loaders, epoch, log_epoch=None):
        if log_epoch is None:
            log_epoch = epoch

        train_loader = train_loaders['train_loader']
        out_distribution_loader = train_loaders['out_distribution_loader']
        self.model.train()

        train_set_batches = self._get_dataloader_length(train_loader, out_distribution_loader=out_distribution_loader)

        # https: // github.com / pytorch / pytorch / issues / 1917  # issuecomment-433698337

        bs = self._get_loader_batchsize(train_loader)
        od_bs = self._get_loader_batchsize(out_distribution_loader)

        clean_loss = self._get_clean_criterion(test=False, log_stats=True, name_prefix='Clean')
        clean_loss, msda = self._get_msda(clean_loss, log_stats=True, name_prefix='Clean')

        od_train_criterion = self._get_od_criterion(epoch, self.model)
        od_train_criterion, od_msda = self._get_msda(od_train_criterion, log_stats=True)

        losses = [clean_loss, od_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        distance_od = DistanceLogger(self.od_distance, name_prefix='OD')
        confidence_od = self._get_od_conf_logger(name_prefix='OD')
        total_loss_logger = SingleValueLogger('Loss')
        lr_logger = SingleValueLogger('LR')
        loggers = [total_loss_logger, acc_conf_clean, confidence_od, distance_od, lr_logger]

        self.output_backend.start_epoch_log(train_set_batches)
        for batch_idx, (clean_data, clean_target) in enumerate(train_loader):
            try:
                od_data, od_target = next(self.od_iterator)
            except:
                self.od_iterator = iter(out_distribution_loader)
                len(out_distribution_loader.dataset)
                od_data, od_target = next(self.od_iterator)

            if clean_data.shape[0] < bs or od_data.shape[0] < od_bs:
                continue

            clean_data = msda(clean_data)
            od_data = od_msda(od_data)

            clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
            od_data, od_target = od_data.to(self.device), od_target.to(self.device)

            od_adv_samples = od_train_criterion.inner_max(od_data, od_target)

            with amp.autocast(enabled=self.mixed_precision):
                loss_closure = self.__get_loss_closure(clean_data, clean_target,
                                   od_adv_samples, od_data, od_target,
                                   clean_loss, od_train_criterion,
                                   total_loss_logger=total_loss_logger,
                                   lr_logger=lr_logger,
                                   acc_conf_clean=acc_conf_clean,
                                   confidence_od=confidence_od,
                                   distance_od=distance_od)


                self._loss_step(loss_closure)
            #ema
            if self.ema:
                self._update_avg_model()

            self._update_scheduler(epoch + (batch_idx + 1) / train_set_batches)
            self.output_backend.log_batch_summary(log_epoch, batch_idx, True, losses=losses, loggers=loggers)

        self._update_scheduler(epoch + 1)
        self.output_backend.end_epoch_write_summary(losses, loggers, log_epoch, True)

    def _update_avg_model_batch_norm(self, train_loaders):
        train_loader = train_loaders['train_loader']
        out_distribution_loader = train_loaders['out_distribution_loader']

        clean_loss = self._get_clean_criterion(test=False, log_stats=True, name_prefix='Clean')
        clean_loss, msda = self._get_msda(clean_loss, log_stats=True, name_prefix='Clean')

        od_train_criterion = self._get_od_criterion(self.epochs, self.avg_model)
        od_train_criterion, od_msda = self._get_msda(od_train_criterion, log_stats=True)
        bs = self._get_loader_batchsize(train_loader)
        od_bs = self._get_loader_batchsize(out_distribution_loader)

        self.avg_model.train()

        with torch.no_grad():
            for batch_idx, (clean_data, clean_target) in enumerate(train_loader):
                try:
                    od_data, od_target = next(self.od_iterator)
                except:
                    self.od_iterator = iter(out_distribution_loader)
                    od_data, od_target = next(self.od_iterator)

                if clean_data.shape[0] < bs or od_data.shape[0] < od_bs:
                    continue

                clean_data = msda(clean_data)
                od_data = od_msda(od_data)

                clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
                od_data, od_target = od_data.to(self.device), od_target.to(self.device)

                od_adv_samples = od_train_criterion.inner_max(od_data, od_target)
                with amp.autocast(enabled=self.mixed_precision):
                    loss_closure = self.__get_loss_closure(clean_data, clean_target,
                                                           od_adv_samples, od_data, od_target,
                                                           clean_loss, od_train_criterion)

                    loss = loss_closure(log=False)
