import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils_svces.distances as d
import math

from .out_distribution_training import OutDistributionTraining
from .helpers import interleave_forward
from .train_loss import CrossEntropyProxy, AccuracyConfidenceLogger, DistanceLogger, ConfidenceLogger, SingleValueLogger, SingleValueHistogramLogger
import torch.cuda.amp as amp
from .in_distribution_training import InDistributionTraining
######################################################

class InOutDistributionTraining(OutDistributionTraining):
    def __init__(self, name, model, id_distance, od_distance, optimizer_config, epochs, device, num_classes,
                 train_clean=True, id_trades=False, id_weight=1.0, clean_weight=1.0, id_adv_weight=1.0,
                 od_trades=False, od_weight=1.0, od_clean_weight=1.0, od_adv_weight=1.0,
                 lr_scheduler_config=None, model_config=None, test_epochs=1, verbose=100, saved_model_dir='SavedModels',
                 saved_log_dir='Logs'):

        super().__init__(name, model, od_distance, optimizer_config, epochs, device, num_classes,
                         lr_scheduler_config=lr_scheduler_config, model_config=model_config, od_weight=od_weight,
                         test_epochs=test_epochs, verbose=verbose, saved_model_dir=saved_model_dir,
                         saved_log_dir=saved_log_dir)

        #ID attributes
        self.train_clean = train_clean
        self.id_trades = id_trades
        self.id_weight = id_weight
        self.clean_weight = clean_weight
        self.id_adv_weight = id_adv_weight
        self.id_distance = id_distance

        #OD attribute
        self.od_trades = od_trades
        self.od_clean_weight = od_clean_weight
        self.od_adv_weight = od_adv_weight

    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
        raise NotImplementedError()

    def _get_od_clean_criterion(self, epoch, model, name_prefix='OD'):
        raise NotImplementedError()

    def _get_id_accuracy_conf_logger(self, name_prefix):
        return AccuracyConfidenceLogger(name_prefix=name_prefix)

    def test(self, test_loaders, epoch, test_avg_model=False):
        if test_avg_model:
            model = self.avg_model
        else:
            model = self.model

        model.eval()

        new_best = False
        if test_avg_model:
            avg_prefix = 'AVG_'
            best_acc = self.best_avg_model_accuracy
        else:
            avg_prefix = ''
            best_acc = self.best_accuracy

        if 'test_loader' in test_loaders:

            test_loader = test_loaders['test_loader']
            id_acc = InDistributionTraining._inner_test(self, model, test_loader,
                                                        epoch, prefix=f'{avg_prefix}Clean', id_prefix=f'{avg_prefix}ID')
            if id_acc > best_acc:
                new_best = True
                if test_avg_model:
                    self.best_avg_model_accuracy = id_acc
                else:
                    self.best_accuracy = id_acc

        if 'extra_test_loaders' in test_loaders:
            for i, test_loader in enumerate(test_loaders['extra_test_loaders']):
                prefix = f'{avg_prefix}CleanExtra{i}'
                id_prefix = f'{avg_prefix}IDExtra{i}'
                InDistributionTraining._inner_test(self, model, test_loader, epoch,
                                                   prefix=prefix, id_prefix=id_prefix)

        return new_best


    def _forward(self, model, clean_data, id_data, clean_od_data, od_data):
        data_list = []
        data_order = [clean_data, id_data, clean_od_data, od_data]
        idcs = []
        idx = 0

        for data in data_order:
            if data is not None:
                data_list.append(data)
                idcs.append(idx)
                idx += 1
            else:
                idcs.append(-1)

        outs = interleave_forward(model, data_list)

        out_list = []
        for idx in idcs:
            if idx == -1:
                out_list.append(None)
            else:
                out_list.append(outs[idx])

        return out_list

    def __get_loss_closure(self, clean_data, clean_target, id_adv_samples, id_data, id_target,
                           clean_od_data, clean_od_target, od_adv_samples, od_data, od_target,
                           clean_loss, id_train_criterion, od_clean_loss, od_train_criterion,
                           total_loss_logger=None,
                           lr_logger=None,
                           acc_conf_clean=None,
                           acc_conf_id=None,
                           distance_id=None,
                           confidence_od=None,
                           distance_od=None):

        def loss_closure(log=False):
            clean_out, id_adv_out, od_clean_out, od_adv_out = \
                self._forward(self.model, clean_data, id_adv_samples, clean_od_data, od_adv_samples)

            # clean loss for clean adv training and trades
            if self.train_clean or self.id_trades:
                loss0 = clean_loss(clean_data, clean_out, clean_data, clean_target)
            else:
                loss0 = torch.tensor(0.0, device=self.device)

            if self.id_trades:
                id_hard_label = clean_target
                id_tar = F.softmax(clean_out, dim=1)
            else:
                id_hard_label = id_target
                id_tar = id_target

            # adversarial loss / trades regularizer
            loss1 = id_train_criterion(id_adv_samples, id_adv_out, id_data, id_tar)

            # od clean loss for od trades
            if self.od_trades:
                od_tar = F.softmax(od_clean_out, dim=1)
                loss2 = od_clean_loss(clean_od_data, od_clean_out, clean_od_data, clean_od_target)
            else:
                od_tar = od_target
                loss2 = torch.tensor(0.0, device=self.device)

            # od acet loss / trades regularizer
            loss3 = od_train_criterion(od_adv_samples, od_adv_out, od_data, od_tar)

            loss = self.id_weight * (self.clean_weight * loss0 + self.id_adv_weight * loss1)
            loss += self.od_weight * (self.od_clean_weight * loss2 + self.od_adv_weight * loss3)

            if log:
                total_loss_logger.log(loss)
                lr_logger.log(self.scheduler.get_last_lr()[0])

                # log
                if self.train_clean or self.id_trades:
                    acc_conf_clean(clean_data, clean_out, clean_data, clean_target)

                acc_conf_id(id_adv_samples, id_adv_out, id_data, id_hard_label)
                distance_id(id_adv_samples, id_adv_out, id_data, id_hard_label)

                confidence_od(od_adv_samples, od_adv_out, od_data, od_target)
                distance_od(od_adv_samples, od_adv_out, od_data, od_target)

            return loss
        return loss_closure

    def _inner_train(self, train_loaders, epoch, log_epoch=None):
        train_loader = train_loaders['train_loader']
        out_distribution_loader = train_loaders['out_distribution_loader']
        self.model.train()

        train_set_batches = self._get_dataloader_length(train_loader, out_distribution_loader=out_distribution_loader)

        # https: // github.com / pytorch / pytorch / issues / 1917  # issuecomment-433698337
        id_iterator = iter(train_loader)
        if self.od_iterator is None:
            self.od_iterator = iter(out_distribution_loader)

        bs = self._get_loader_batchsize(train_loader)
        od_bs = self._get_loader_batchsize(out_distribution_loader)
        if od_bs != bs:
            raise AssertionError('Out distribution and in distribution cifar_loader need to have the same batchsize')

        clean_loss = self._get_clean_criterion(name_prefix='Clean')
        id_train_criterion = self._get_id_criterion(epoch, self.model, name_prefix='ID')
        od_clean_loss = self._get_od_clean_criterion(epoch, self.model, name_prefix='OD')
        od_train_criterion = self._get_od_criterion(epoch, self.model, name_prefix='OD')
        losses = [clean_loss, id_train_criterion, od_clean_loss, od_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_id = self._get_id_accuracy_conf_logger(name_prefix='ID')
        distance_id = DistanceLogger(self.id_distance, name_prefix='ID')

        confidence_od = self._get_od_conf_logger(name_prefix='OD')
        distance_od = DistanceLogger(self.od_distance, name_prefix='OD')
        total_loss_logger = SingleValueLogger('Loss')
        lr_logger = SingleValueLogger('LR')
        loggers = [total_loss_logger, acc_conf_id, distance_id, acc_conf_clean, confidence_od, distance_od, lr_logger]

        self.output_backend.start_epoch_log(train_set_batches)

        for batch_idx, (id_data, id_target) in enumerate(id_iterator):
            #sample clean ref_data

            id_data, id_target = id_data.to(self.device), id_target.to(self.device)

            if self.train_clean:
                #if train clean, sample new clean data
                try:
                    clean_data, clean_target = next(id_iterator)
                    clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
                except StopIteration:
                    break
            elif self.id_trades:
                #if id trades, repeat clean data for trades regularizer
                clean_data = id_data.detach().clone()
                clean_target = id_target.detach().clone()
            else:
                #else, no clean data needed
                clean_data = None
                clean_target = None

            #sample od ref_data
            try:
                od_data, od_target = next(self.od_iterator)
            except StopIteration:
                #when od iterator runs out, jsut start from beginning
                self.od_iterator = iter(out_distribution_loader)
                od_data, od_target = next(self.od_iterator)

            od_data, od_target = od_data.to(self.device), od_target.to(self.device)

            if self.od_trades:
                #if od trades, repeat od data for trades od regularizer
                clean_od_data = id_data.detach().clone()
                clean_od_target = id_target.detach().clone()
            else:
                #else, no clean od data needed
                clean_od_data = None
                clean_od_target = None

            if (id_data.shape[0] < bs) or (od_data.shape[0] < bs) or (self.train_clean and clean_data.shape[0] < bs):
                continue

            #id_attack
            id_adv_samples = id_train_criterion.inner_max(id_data, id_target)

            #od attack
            od_adv_samples = od_train_criterion.inner_max(od_data, od_target)

            with amp.autocast(enabled=self.mixed_precision):
                loss_closure = self.__get_loss_closure(clean_data, clean_target, id_adv_samples, id_data, id_target,
                                                       clean_od_data, clean_od_target, od_adv_samples, od_data, od_target,
                                                       clean_loss, id_train_criterion, od_clean_loss, od_train_criterion,
                                                       total_loss_logger=total_loss_logger,
                                                       lr_logger=lr_logger,
                                                       acc_conf_clean=acc_conf_clean,
                                                       acc_conf_id=acc_conf_id,
                                                       distance_id=distance_id,
                                                       confidence_od=confidence_od,
                                                       distance_od=distance_od)


                self._loss_step(loss_closure)

            #ema
            if self.ema:
                self._update_avg_model()

            self._update_scheduler(epoch + (batch_idx + 1) / train_set_batches)
            self.output_backend.log_batch_summary(epoch, batch_idx, True, losses=losses, loggers=loggers)

        self._update_scheduler(epoch + 1)
        self.output_backend.end_epoch_write_summary(losses, loggers, epoch, True)

    def _update_avg_model_batch_norm(self, train_loaders):
        train_loader = train_loaders['train_loader']
        out_distribution_loader = train_loaders['out_distribution_loader']

        id_iterator = iter(train_loader)

        clean_loss = self._get_clean_criterion()
        id_train_criterion = self._get_id_criterion(0, self.avg_model)
        od_clean_loss = self._get_od_clean_criterion(epoch, self.model, name_prefix='OD')
        od_train_criterion = self._get_od_criterion(0, self.avg_model)
        bs = self._get_loader_batchsize(train_loader)
        od_bs = self._get_loader_batchsize(out_distribution_loader)

        self.avg_model.train()

        with torch.no_grad():
            for batch_idx, (id_data, id_target) in enumerate(id_iterator):
                # sample clean ref_data

                id_data, id_target = id_data.to(self.device), id_target.to(self.device)

                if self.train_clean:
                    # if train clean, sample new clean data
                    try:
                        clean_data, clean_target = next(id_iterator)
                        clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
                    except StopIteration:
                        break
                elif self.id_trades:
                    # if id trades, repeat clean data for trades regularizer
                    clean_data = id_data.detach().clone()
                    clean_target = id_target.detach().clone()
                else:
                    # else, no clean data needed
                    clean_data = None
                    clean_target = None

                # sample od ref_data
                try:
                    od_data, od_target = next(self.od_iterator)
                except StopIteration:
                    # when od iterator runs out, jsut start from beginning
                    self.od_iterator = iter(out_distribution_loader)
                    od_data, od_target = next(self.od_iterator)

                if self.od_trades:
                    # if od trades, repeat od data for trades od regularizer
                    clean_od_data = id_data.detach().clone()
                    clean_od_target = id_target.detach().clone()
                else:
                    # else, no clean od data needed
                    clean_od_data = None
                    clean_od_target = None

                if (id_data.shape[0] < bs) or (od_data.shape[0] < od_bs) or (
                        self.train_clean and clean_data.shape[0] < bs):
                    continue

                od_data, od_target = od_data.to(self.device), od_target.to(self.device)

                # id_attack
                id_adv_samples = id_train_criterion.inner_max(id_data, id_target)

                # od attack
                od_adv_samples = od_train_criterion.inner_max(od_data, od_target)

                with amp.autocast(enabled=self.mixed_precision):
                    loss_closure = self.__get_loss_closure(clean_data, clean_target, id_adv_samples, id_data, id_target,
                                                           clean_od_data, clean_od_target, od_adv_samples, od_data,
                                                           od_target,
                                                           clean_loss, id_train_criterion, od_clean_loss,
                                                           od_train_criterion)

                    loss_closure(log=False)
