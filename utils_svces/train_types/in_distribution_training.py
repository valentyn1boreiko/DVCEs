import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_type import TrainType
from .train_loss import AccuracyConfidenceLogger, DistanceLogger, SingleValueLogger

from .helpers import interleave_forward
import torch.cuda.amp as amp

#Base class for train types that use custom losses/attacks on the in distribution such as adversarial training
class InDistributionTraining(TrainType):
    def __init__(self, name, model, id_distance, optimizer_config, epochs, device, num_classes,
                 clean_criterion='ce', train_clean=True, id_trades=False, clean_weight=1.0, id_adv_weight=1.0,
                 lr_scheduler_config=None, msda_config=None, model_config=None,
                 test_epochs=1, verbose=100, saved_model_dir='SavedModels', saved_log_dir='Logs'):
        super().__init__(name, model, optimizer_config, epochs, device, num_classes,
                         clean_criterion=clean_criterion,
                         lr_scheduler_config=lr_scheduler_config, msda_config=msda_config,
                         model_config=model_config, test_epochs=test_epochs,
                         verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

        self.train_clean = train_clean
        self.id_trades = id_trades
        self.clean_weight = clean_weight
        self.id_adv_weight = id_adv_weight
        self.id_distance = id_distance

    def _get_id_criterion(self, epoch, model, name_prefix='ID'):
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
            id_acc = self._inner_test(model, test_loader, epoch, prefix=f'{avg_prefix}Clean', id_prefix=f'{avg_prefix}ID')
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
                self._inner_test(model, test_loader, epoch, prefix=prefix, id_prefix=id_prefix)

        return new_best

    def _inner_test(self, model, test_loader, epoch, prefix='Clean', id_prefix='ID', *args, **kwargs):
        test_set_batches = len(test_loader)
        clean_loss = self._get_clean_criterion(test=True, log_stats=True, name_prefix=prefix)

        id_train_criterion = self._get_id_criterion(0, model,
                                                    name_prefix=id_prefix)  #set 0 as epoch so it uses same attack steps every time
        losses = [clean_loss, id_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix=prefix)
        acc_conf_adv = self._get_id_accuracy_conf_logger(name_prefix=id_prefix)
        distance_adv = DistanceLogger(self.id_distance, name_prefix=id_prefix)
        loggers = [acc_conf_clean, acc_conf_adv, distance_adv]

        self.output_backend.start_epoch_log(test_set_batches)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                adv_samples = id_train_criterion.inner_max(data, target)
                clean_out, adv_out = interleave_forward(model, [data, adv_samples])

                if self.id_trades:
                    id_target = F.softmax(clean_out, dim=1)
                else:
                    id_target = target

                loss0 = clean_loss(data, clean_out, data, target)
                loss1 = id_train_criterion(adv_samples, adv_out, data, id_target)

                acc_conf_clean(data, clean_out, data, target)
                acc_conf_adv(adv_samples, adv_out, data, target)
                distance_adv(adv_samples, adv_out, data, target)
                self.output_backend.log_batch_summary(epoch, batch_idx, False, losses=losses, loggers=loggers)

        self.output_backend.end_epoch_write_summary(losses, loggers, epoch, False)
        id_acc = acc_conf_adv.get_accuracy()
        return id_acc

    def __get_loss_closure(self, clean_data, clean_target, id_adv_samples, id_data, id_target,
                           clean_loss, id_train_criterion,
                           total_loss_logger=None,
                           lr_logger=None,
                           acc_conf_clean=None,
                           acc_conf_adv=None,
                           distance_adv=None
                           ):

        def loss_closure(log=False):
            if self.train_clean or self.id_trades:
                clean_out, adv_out = interleave_forward(self.model, [clean_data, id_adv_samples])

                if self.id_trades:
                    id_hard_label = clean_target
                    id_tar = F.softmax(clean_out, dim=1)
                else:
                    id_hard_label = id_target
                    id_tar = id_target

                loss0 = clean_loss(clean_data, clean_out, clean_data, clean_target)
                loss1 = id_train_criterion(id_adv_samples, adv_out, id_data, id_tar)
                loss = self.clean_weight * loss0 + self.id_adv_weight * loss1
            else:
                id_hard_label = id_target
                adv_out = self.model(id_adv_samples)
                clean_out = None
                loss = id_train_criterion(id_adv_samples, adv_out, id_data, id_target)

            if log:
                total_loss_logger.log(loss)
                lr_logger.log(self.scheduler.get_last_lr()[0])

                # log
                if self.train_clean or self.id_trades:
                    acc_conf_clean(clean_data, clean_out, clean_data, clean_target)

                acc_conf_adv(id_adv_samples, adv_out, id_data, id_hard_label)
                distance_adv(id_adv_samples, adv_out, id_data, id_hard_label)
            return loss

        return loss_closure

    def _inner_train(self, train_loaders, epoch, log_epoch=None):
        if log_epoch is None:
            log_epoch = epoch

        self.model.train()
        train_loader = train_loaders['train_loader']

        train_set_batches = self._get_dataloader_length(train_loader)
        bs = self._get_loader_batchsize(train_loader)

        clean_loss = self._get_clean_criterion(log_stats=True, name_prefix='Clean')
        id_train_criterion = self._get_id_criterion(epoch, self.model)
        losses = [clean_loss, id_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_adv = self._get_id_accuracy_conf_logger(name_prefix='ID')
        distance_adv = DistanceLogger(self.id_distance, name_prefix='ID')
        total_loss_logger = SingleValueLogger('Loss')
        lr_logger = SingleValueLogger('LR')
        loggers = [total_loss_logger, acc_conf_clean, acc_conf_adv, distance_adv, lr_logger]

        id_iterator = iter(train_loader)

        self.output_backend.start_epoch_log(train_set_batches)

        for batch_idx, (id_data, id_target) in enumerate(id_iterator):

            id_data, id_target = id_data.to(self.device), id_target.to(self.device)
            # sample clean ref_data
            if self.train_clean:
                try:
                    clean_data, clean_target = next(id_iterator)
                    clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
                except StopIteration:
                    break
            elif self.id_trades:
                clean_data = id_data.detach().clone()
                clean_target = id_target.detach().clone()
            else:
                clean_data = None
                clean_target = None

            if id_data.shape[0] < bs or (self.train_clean and clean_data.shape[0] < bs):
                continue

            id_adv_samples = id_train_criterion.inner_max(id_data, id_target)
            with amp.autocast(enabled=self.mixed_precision):
                loss_closure = self.__get_loss_closure(clean_data, clean_target, id_adv_samples, id_data, id_target,
                                                       clean_loss, id_train_criterion,
                                                       total_loss_logger=total_loss_logger,
                                                       lr_logger=lr_logger,
                                                       acc_conf_clean=acc_conf_clean,
                                                       acc_conf_adv=acc_conf_adv,
                                                       distance_adv=distance_adv
                                                       )

                self._loss_step(loss_closure)

            #ema
            if self.ema:
                 self._update_avg_model()

            self._update_scheduler(epoch + (batch_idx + 1) / train_set_batches)
            self.output_backend.log_batch_summary(log_epoch, batch_idx, True, losses=losses, loggers=loggers)

        self._update_scheduler(epoch + 1)
        self.output_backend.end_epoch_write_summary(losses, loggers, log_epoch, True)

    def _update_avg_model_batch_norm(self, train_loaders):
        self.avg_model.train()
        train_loader = train_loaders['train_loader']

        train_set_batches = self._get_dataloader_length(train_loader)
        bs = self._get_loader_batchsize(train_loader)

        clean_loss = self._get_clean_criterion(log_stats=True, name_prefix='Clean')
        id_train_criterion = self._get_id_criterion(0, self.avg_model)
        id_iterator = iter(train_loader)

        self.output_backend.start_epoch_log(train_set_batches)

        with torch.no_grad():
            for batch_idx, (id_data, id_target) in enumerate(id_iterator):
                # sample clean ref_data

                id_data, id_target = id_data.to(self.device), id_target.to(self.device)
                if self.train_clean:
                    try:
                        clean_data, clean_target = next(id_iterator)
                        clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
                    except StopIteration:
                        break
                elif self.id_trades:
                    clean_data = id_data.detach().clone()
                    clean_target = id_target.detach().clone()
                else:
                    clean_data = None
                    clean_target = None

                if id_data.shape[0] < bs or (self.train_clean and clean_data.shape[0] < bs):
                        continue

                id_adv_samples = id_train_criterion.inner_max(id_data, id_target)
                with amp.autocast(enabled=self.mixed_precision):
                    loss_closure = self.__get_loss_closure(clean_data, clean_target, id_adv_samples, id_data, id_target,
                                                           clean_loss, id_train_criterion)

                    loss_closure(log=False)

