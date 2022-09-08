from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEFAULT_LENGTH = 50000

def double_array(array):
    if len(array.shape) == 1:
        new_shape = (2 * array.shape[0],)
        temp = array.new_empty(new_shape)
        temp[:array.shape[0]] = array
    else:
        new_shape = (2 * array.shape[0],) + array.shape[1:]
        temp = array.new_empty(new_shape)
        temp[:array.shape[0],:] = array

    return temp

class LogType(Enum):
    SCALAR = auto()
    HISTOGRAM = auto()


#return type of Loggers and LoggingLosses
class Log():
    def __init__(self, name, value, type):
        if not isinstance(type, LogType):
            raise ValueError('Log expects LogType as type')

        self.name = name
        self.value = value
        self.type = type

class Logger():
    def __int__(self, name, name_prefix=None):
        self.name = name
        self.name_prefix = name_prefix
        self._reset_stats()

    def _reset_stats(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def get_logs(self):
        #return list of Logs
        return []

class SingleValueLogger(Logger):
    def __init__(self, name, name_prefix=None):
        super().__int__(name, name_prefix=name_prefix)

    def _reset_stats(self):
        self.value = RunningAverage()

    def log(self, value):
        self.value.add_value(value)

    def get_logs(self):
        if self.value.N > 0:
            if self.name_prefix is None:
                tag = self.name
            else:
                tag = f'{self.name_prefix}{self.name}'

            log = Log(tag, self.value.mean, LogType.SCALAR)
            return [log]
        else:
            return []

    def get_all_recorded_values(self):
        return self.values[:self.idx]

class SingleValueHistogramLogger(Logger):
    def __init__(self, name, num_batches=None, name_prefix=None):
        self.num_batches = num_batches
        super().__int__(name, name_prefix=name_prefix)

    def _reset_stats(self):
        self.idx = 0
        if self.num_batches is not None:
            self.values = torch.zeros(self.num_batches)
        else:
            self.values = torch.zeros(DEFAULT_LENGTH)

    def log(self, value):
        if self.idx >= self.values.shape[0]:
            self.values = double_array(self.values)

        self.values[self.idx] = value
        self.idx += 1

    def get_logs(self):
        if self.idx > 0:
            if self.name_prefix is None:
                tag = self.name
            else:
                tag = f'{self.name_prefix}{self.name}'


            log = Log(tag, self.values[:self.idx], LogType.HISTOGRAM)

            return [log]
        else:
            return []

    def get_all_recorded_values(self):
        return self.values[:self.idx]

#Logger subclass that shares the callable interface with TrainLoss
class CallableLogger(Logger):
    def __init__(self, name, name_prefix):
        super().__int__(name, name_prefix=name_prefix)

    def __call__(self, data, model_out, orig_data, y):
        return self.log(data, model_out, orig_data, y)

    def log(self, data, model_out, orig_data, y):
        pass

class RunningAverage():
    def __init__(self):
        self._reset_stats()

    def _reset_stats(self):
        self.N = 0
        self.mean = 0.

    def add_value(self, values_sum, values_N=1):
        new_N = self.N + values_N
        self.mean = self.mean * (self.N / new_N) + values_sum / (new_N)
        self.N = new_N

    def __format__(self, format_spec):
        return format(self.mean, format_spec)

#track accuract and confidence
class AccuracyConfidenceLogger(CallableLogger):
    def __init__(self, name_prefix=None):
        super().__int__('AccuracyConfidence', name_prefix=name_prefix)

    def _reset_stats(self):
        self.accuracy = RunningAverage()
        self.avg_max_confidence = RunningAverage()

    def log(self, data, model_out, orig_data, y):
        conf, predicted = F.softmax(model_out, dim=1).max(dim=1)
        if y.dim() == 1:
            y_tar = y
        elif y.dim() == 2:
            #soft labels
            _, y_tar = y.max(dim=1)
        else:
            raise ValueError()

        correct = predicted.eq(y_tar)
        self.accuracy.add_value(correct.sum().item(), correct.shape[0])
        self.avg_max_confidence.add_value(conf.sum().item(), correct.shape[0])

    def get_accuracy(self):
        return self.accuracy.mean

    def get_logs(self):
        if self.accuracy.N > 0:
            # (name, value, type)
            if self.name_prefix is None:
                acc_name = 'Accuracy'
                conf_name = 'MeanMaxConf'
            else:
                acc_name = f'{self.name_prefix}Accuracy'
                conf_name = f'{self.name_prefix}MeanMaxConf'

            acc_log = Log(acc_name, self.accuracy.mean, LogType.SCALAR)
            conf_log = Log(conf_name, self.avg_max_confidence.mean, LogType.SCALAR)

            return [acc_log, conf_log]
        else:
            return []


class BCAccuracyConfidenceLogger(CallableLogger):
    def __init__(self, num_attributes, name_prefix=None):
        super().__int__('BCAccuracyConfidence', name_prefix=name_prefix)
        self.num_attributes = num_attributes

    def _reset_stats(self):
        self.attribute_accuracies = RunningAverage()
        self.attribute_avg_max_confidences = RunningAverage()

    def get_accuracy(self):
        return torch.mean(self.attribute_accuracies.mean)

    def log(self, data, model_out, orig_data, y):
        bs = data.shape[0]
        sigmoid_attributes = torch.sigmoid(model_out)
        predicted_bool = (sigmoid_attributes > 0.5)
        conf_attributes = torch.zeros_like(model_out)
        conf_attributes[predicted_bool] = sigmoid_attributes[predicted_bool]
        conf_attributes[~predicted_bool] = 1.0 - sigmoid_attributes[~predicted_bool]
        conf_attributes_sum = torch.sum(conf_attributes, dim=0).detach().cpu()
        correct_all = predicted_bool.eq(y)
        correct_per_attribute = torch.sum(correct_all, dim=0).float().detach().cpu()
        self.attribute_accuracies.add_value(correct_per_attribute, bs)
        self.attribute_avg_max_confidences.add_value(conf_attributes_sum, bs)

    def get_logs(self):
        if self.attribute_accuracies.N > 0:
            # (name, value, type)
            if self.name_prefix is None:
                acc_name = 'Accuracy'
                acc_histogram_name = 'IndividualAccuracies'
                conf_name = 'MeanMaxConf'
                conf_histogram_name = 'IndividualAccuraciesMeanMaxConf'
            else:
                acc_name = f'{self.name_prefix}Accuracy'
                acc_histogram_name = f'{self.name_prefix}IndividualAccuracies'
                conf_name = f'{self.name_prefix}MeanMaxConf'
                conf_histogram_name = f'{self.name_prefix}IndividualAccuraciesMeanMaxConf'

            acc_log = Log(acc_name, self.attribute_accuracies.mean.mean(), LogType.SCALAR)
            indiv_accs_log = Log(acc_histogram_name, self.attribute_accuracies.mean, LogType.HISTOGRAM)
            conf_log = Log(conf_name, self.attribute_avg_max_confidences.mean.mean(), LogType.SCALAR)
            indiv_confs_log = Log(conf_histogram_name, self.attribute_avg_max_confidences.mean, LogType.HISTOGRAM)
            return [acc_log, indiv_accs_log, conf_log, indiv_confs_log]
        else:
            return []


class ConfidenceLogger(CallableLogger):
    def __init__(self, name_prefix=None):
        super().__int__('Confidence', name_prefix=name_prefix)

    def _reset_stats(self):
        self.avg_max_confidence = RunningAverage()

    def log(self, data, model_out, orig_data, y):
        conf, predicted = F.softmax(model_out, dim=1).max(dim=1)
        self.avg_max_confidence.add_value(conf.sum().item(), data.shape[0])

    def get_logs(self):
        if self.avg_max_confidence.N > 0:

            # (name, value, type)
            if self.name_prefix is None:
                conf_name = 'MeanMaxConf'
            else:
                conf_name = f'{self.name_prefix}MeanMaxConf'

            conf_log = Log(conf_name, self.avg_max_confidence.mean, LogType.SCALAR)

            return [conf_log]
        else:
            return []

class BCConfidenceLogger(CallableLogger):
    def __init__(self, num_attributes, name_prefix=None):
        super().__int__('BCConfidence', name_prefix=name_prefix)
        self.num_attributes = num_attributes

    def _reset_stats(self):
        self.attribute_avg_max_confidences = RunningAverage()

    def log(self, data, model_out, orig_data, y):
        bs = data.shape[0]
        sigmoid_attributes = torch.sigmoid(model_out)
        predicted_bool = (sigmoid_attributes > 0.5)
        conf_attributes = torch.zeros_like(model_out)
        conf_attributes[predicted_bool] = sigmoid_attributes[predicted_bool]
        conf_attributes[~predicted_bool] = 1.0 - sigmoid_attributes[~predicted_bool]
        conf_attributes_sum = torch.sum(conf_attributes, dim=0).detach().cpu()
        self.attribute_avg_max_confidences.add_value(conf_attributes_sum, bs)

    def get_logs(self):
        if self.attribute_avg_max_confidences.N > 0:
            # (name, value, type)
            if self.name_prefix is None:
                conf_name = 'MeanMaxConf'
                conf_histogram_name = 'IndividualAccuraciesMeanMaxConf'
            else:
                conf_name = f'{self.name_prefix}MeanMaxConf'
                conf_histogram_name = f'{self.name_prefix}IndividualAccuraciesMeanMaxConf'

            conf_log = Log(conf_name, self.attribute_avg_max_confidences.mean.mean(), LogType.SCALAR)
            indiv_confs_log = Log(conf_histogram_name, self.attribute_avg_max_confidences.mean, LogType.HISTOGRAM)
            return [conf_log, indiv_confs_log]
        else:
            return []

#track accuracy and confidence
class DistanceHistogramLogger(CallableLogger):
    def __init__(self, distance,  number_of_datapoints=None, name_prefix=None):
        self.distance = distance
        self.number_of_datapoints = number_of_datapoints
        super().__int__('DistanceHistogramLogger', name_prefix=name_prefix)

    def _reset_stats(self):
        self.idx = 0
        if self.number_of_datapoints is not None:
            self.distances = torch.zeros(self.number_of_datapoints)
        else:
            self.distances = torch.zeros(DEFAULT_LENGTH)

    def log(self, data, model_out, orig_data, y):
        d = self.distance(data, orig_data)
        free_space = (self.distances.shape[0] - self.idx) - 1
        while d.shape[0] >= free_space:
            # double space
            self.distances = double_array(self.distances)
            free_space = (self.distances.shape[0] - self.idx) - 1

        new_idx = self.idx + d.shape[0]
        self.distances[self.idx:new_idx] = d
        self.idx = new_idx

    def get_logs(self):
        # (name, value, type)
        distances_filled = self.distances[:self.idx]
        if self.idx > 0:
            if self.name_prefix is None:
                d_name = 'Distance'
            else:
                d_name = f'{self.name_prefix}Distance'

            log_hist = Log(d_name, distances_filled, LogType.HISTOGRAM)

            if self.name_prefix is None:
                d_name = 'MeanDistance'
            else:
                d_name = f'{self.name_prefix}MeanDistance'

            log_mean = Log(d_name, torch.mean(distances_filled), LogType.SCALAR)

            if self.name_prefix is None:
                d_name = 'VarianceDistance'
            else:
                d_name = f'{self.name_prefix}VarianceDistance'

            log_var = Log(d_name, torch.var(distances_filled), LogType.SCALAR)


            return [log_hist, log_mean, log_var]
        else:
            return []

class DistanceLogger(CallableLogger):
    def __init__(self, distance,  number_of_datapoints=None, name_prefix=None):
        self.distance = distance
        super().__int__('DistanceLogger', name_prefix=name_prefix)

    def _reset_stats(self):
        self.mean_distance = RunningAverage()

    def log(self, data, model_out, orig_data, y):
        d = self.distance(data, orig_data)
        self.mean_distance.add_value(d.sum().item(), d.shape[0])


    def get_logs(self):
        # (name, value, type)
        if self.mean_distance.N > 0:
            if self.name_prefix is None:
                d_name = 'MeanDistance'
            else:
                d_name = f'{self.name_prefix}MeanDistance'

            log_mean = Log(d_name, self.mean_distance.mean, LogType.SCALAR)

            return [log_mean]
        else:
            return []


class TrainLoss(nn.Module):
    def __init__(self, name, expected_format='log_probabilities'):
        super().__init__()
        self.name = name
        self.expected_format = expected_format

    def get_config(self):
        return {'name': self.name}

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        raise NotImplementedError()

    def get_logs(self):
        return []

    def _prepare_input(self, model_out):
        #density_model out are logits
        if self.expected_format == 'log_probabilities':
            out = torch.nn.functional.log_softmax(model_out, dim=1)
        elif self.expected_format == 'probabilities':
            out = torch.nn.functional.softmax(model_out, dim=1)
        elif self.expected_format == 'logits':
            out = model_out
        else:
            raise ValueError(f'Format {self.expected_format} not supported')
        return out

    @staticmethod
    def reduce(loss, reduction) :
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError('reduction not supported')

#log the mean loss
#supports multiple sub losses
class LoggingLoss(TrainLoss):
    def __init__(self, name, expected_format, num_losses=1, sub_losses_postfix=None, log_stats=False, name_prefix=None):
        TrainLoss.__init__(self, name, expected_format=expected_format)

        self.log_stats = log_stats
        self.name_prefix = name_prefix
        self.num_Losses = num_losses
        self.sub_losses_postfix = sub_losses_postfix

        self._reset_stats()

    def set_log(self, log):
        self.log_stats = log

    def _reset_stats(self):
        self.loss_means = []
        for i in range(self.num_Losses):
            self.loss_means.append(RunningAverage())

    def _log_stats(self, loss_expanded, loss_idx=0):
        if self.log_stats:
            self.loss_means[loss_idx].add_value(loss_expanded.sum().item(), loss_expanded.shape[0])

    def get_logs(self):
        logs = []

        if self.loss_means[0].N > 0:
            for i in range(self.num_Losses):
                # scalar epoch wide mean loss
                if self.name_prefix is None:
                    full_name = self.name
                else:
                    full_name = f'{self.name_prefix}{self.name}'

                if self.sub_losses_postfix is not None:
                    full_name = f'{full_name}_{self.sub_losses_postfix[i]}'
                elif self.num_Losses > 1:
                    full_name = f'{full_name}_{i}'

                logs.append(Log(full_name, self.loss_means[i].mean, LogType.SCALAR))

        return logs

class MinMaxLoss(LoggingLoss):
    def __init__(self, name, expected_format, log_stats=False, num_losses=1, sub_losses_postfix=None, name_prefix=None):
        super().__init__(name, expected_format, num_losses=num_losses, sub_losses_postfix=sub_losses_postfix,
                         log_stats=log_stats, name_prefix=name_prefix)

    def inner_max(self, data, target):
        raise NotImplementedError()

class ZeroLoss(MinMaxLoss):
    def __init__(self, log_stats=False, name_prefix=None):
        super().__init__('EmptyLoss', expected_format='logits', log_stats=log_stats, name_prefix=name_prefix)

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        loss_expanded = torch.zeros(data.shape[0], dtype=torch.float, device=data.device)
        return TrainLoss.reduce(loss_expanded, reduction)

    def inner_max(self, data, target):
        return data

class CrossEntropyProxy(LoggingLoss):
    def __init__(self, log_stats=False, name_prefix=None):
        super().__init__('CrossEntropy', expected_format='logits', log_stats=log_stats, name_prefix=name_prefix)

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = F.cross_entropy(prep_out, y, reduction='none' )
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class BCELogitsProxy(LoggingLoss):
    def __init__(self, mask=None, log_stats=False, name_prefix=None):
        super().__init__('BCELogits', expected_format='logits', log_stats=log_stats, name_prefix=name_prefix)
        self.mask = mask

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = F.binary_cross_entropy_with_logits(prep_out, y.float(), reduction='none' )
        if self.mask is None:
            loss_expanded = loss_expanded.mean(dim=1)
        else:
            nnz_attributes = self.mask.sum(dim=1)
            loss_expanded = (loss_expanded * self.mask).sum(dim=1) / nnz_attributes
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class NLLProxy(LoggingLoss):
    def __init__(self, log_stats=False, name_prefix=None):
        super().__init__('NLLoss', expected_format='log_probabilities', log_stats=log_stats, name_prefix=name_prefix)

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = F.nll_loss(prep_out, y, reduction='none' )
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class KLDivergenceProxy(LoggingLoss):
    def __init__(self, log_stats=False, name_prefix=None):
        super().__init__('KLDivergence', expected_format='log_probabilities', log_stats=log_stats,
                         name_prefix=name_prefix)

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        #unlike other losses, KL dvergence expects y to be in the one-hot format
        prep_out = self._prepare_input(model_out)
        loss_expanded = F.kl_div(prep_out, y, reduction='none' ).sum(dim=1)
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

#calculate entropy of NxD matrix, where each of the N rows corresponds to a D-dimensional categorical distribution
def entropy(logits):
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    return -torch.sum(probabilities * log_probabilities, dim=1)

class KLDivergenceEntropyMinimizationProxy(LoggingLoss):
    def __init__(self, entropy_weight=1, log_stats=False, name_prefix=None):
        super().__init__('KLDivergence', expected_format='log_probabilities', num_losses=2,
                         sub_losses_postfix=['', 'Entropy'], log_stats=log_stats, name_prefix=name_prefix)
        self.entropy_weight = entropy_weight

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        #unlike other losses, KL dvergence expects y to be in the one-hot format
        prep_out = self._prepare_input(model_out)
        kl_expanded = F.kl_div(prep_out, y, reduction='none' ).sum(dim=1)
        entropy_expanded = entropy(model_out)
        loss_expanded = kl_expanded + self.entropy_weight * entropy_expanded
        self._log_stats(kl_expanded, loss_idx=0)
        self._log_stats(entropy_expanded, loss_idx=1)
        return TrainLoss.reduce(loss_expanded, reduction)

class ConfidenceLoss(LoggingLoss):
    def __init__(self, log_stats=False, name_prefix=None):
        super().__init__('ConfidenceLoss', expected_format='probabilities', log_stats=log_stats,
                         name_prefix=name_prefix)

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = prep_out[torch.arange(0, prep_out.shape[0]), y]
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class LogConfidenceLoss(LoggingLoss):
    def __init__(self, log_stats=False, name_prefix=None):
        super().__init__('ConfidenceLoss', expected_format='log_probabilities', log_stats=log_stats,
                         name_prefix=name_prefix)

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = prep_out[torch.arange(0, prep_out.shape[0]), y]
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class MaxConfidenceLoss(LoggingLoss):
    def __init__(self, log_stats=False, name_prefix=None):
        super().__init__('ConfidenceLoss', expected_format='probabilities', log_stats=log_stats,
                         name_prefix=name_prefix)

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = torch.max(prep_out, dim=1)[0]
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

class LossWrapper(TrainLoss):
    def __init__(self, wrapper_name, train_loss):
        super().__init__(f'{wrapper_name}_{train_loss.name}', expected_format='logits')
        self.train_loss = train_loss

    def get_config(self):
        return self.train_loss.get_config()

    def get_logs(self):
        return self.train_loss.get_logs()

class NegativeWrapper(LossWrapper):
    def __init__(self, train_loss):
        super().__init__('NegativeWrapper', train_loss)

    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        out = - self.train_loss(data, model_out, orig_data, y, reduction=reduction)
        return out

def acet_target_obj_from_name(obj_name):
    if obj_name in ['kl', 'KL']:
        return lambda x, y: kl(x, y), 'log_probabilities'
    else:
        raise NotImplementedError()

#out of distribution objectives for acet
#returns objective, epxected_format
def acet_uniform_obj_from_name(obj_name, K):
    if obj_name == 'conf':
        return conf, 'probabilities'
    if obj_name == 'logit_diff':
        return max_diffs, 'logits'
    if obj_name == 'conf_diff':
        return max_diffs, 'probabilities'
    if obj_name == 'norm_conf':
        return lambda x: normalized_conf(x, K), 'probabilities'
    elif obj_name == 'log_conf':
        return log_conf, 'log_probabilities'
    elif obj_name == 'neg_entropy':
        return neg_entropy, 'logits'
    elif obj_name == 'sqr_conf':
        return lambda x: sqr_conf(x, K), 'probabilities'
    elif obj_name == 'minus_log_minus_conf':
        return lambda x: minus_log_minus_conf(x, K), 'probabilities'
    elif obj_name in ['kl', 'KL']:
        return lambda x: kl(x, (1/K) * torch.ones_like(x)), 'log_probabilities'
    elif obj_name == 'bhattacharyya':
        return lambda x: bhattacharya(x, (1/K) * torch.ones_like(x)), 'probabilities'
    elif 'renyi_' in obj_name:
        alpha_str = obj_name[6:]
        if alpha_str == 'inf':
            return lambda  x: renyi_inf(x, (1/K) * torch.ones_like(x)), 'probabilities'
        else:
            alpha = float(alpha_str)
            return lambda  x: renyi_divergence(x.exp(), (1 / K) * torch.ones_like(x), alpha), 'probabilities'
    elif obj_name == 'max_conf_logits':
        return max_conf_logits, 'logits'
    else:
        raise ValueError('Objective {} is not supported'.format(obj_name))

def renyi_divergence(p, q, alpha, p_eps=0, q_eps=0):
    return (1 / (alpha - 1.)) * torch.log( torch.sum( (p+p_eps).pow(alpha) / (q+q_eps).pow(alpha-1.), dim=1))

def renyi_inf(p, q):
    return torch.log( torch.max(p / q,dim=1)[0])

def max_conf_logits(logits):
    max_logits = logits.max(dim=1)[0]
    other_logits_sum = torch.sum(logits,dim=1) - max_logits
    return max_logits - other_logits_sum


def max_diffs(logits):
    max, _ = logits.max(dim=1)
    min, _ = logits.min(dim=1)
    return max - min

def kl(model_out, target):
    KL = torch.nn.functional.kl_div(model_out, target, reduction='none').sum(dim=1)
    return KL

def bhattacharya(p,q, eps=1e-8):
    #p,q probabilitie distributions over dim 1!
    BC = torch.sum(torch.sqrt(p * q + eps), dim=1)
    d = -torch.log(BC)
    return d

def sqr_conf(p, K):
    return (p.max(1)[0] - 1 / K) ** 2

def minus_log_minus_conf(p, K):
    # this is the convex equivalent of using confs,
    # advantages: gets flatter as max_conf approaches 1/k and is 0 at 1/k
    return -torch.log(1 + 1 / K - p.max(1)[0])  # -ln( 1 - max(softmax out) + 1/k)

def normalized_conf(p, K):
    return (p.max(1)[0] - 1 / K) / (1 - 1 / K)

def conf(p):
    return p.max(1)[0]

def log_conf(p):
    return p.max(1)[0]

def neg_entropy(logits):
    #most stable version of calculating this
    return torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
