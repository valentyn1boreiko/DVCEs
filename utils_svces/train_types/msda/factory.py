from .fmix import FMix
from .mixup import Mixup
from .dummy_msda import DummyMSDA

def get_msda(loss, msda_config, log_stats=True, name_prefix=None):
    if msda_config is None:
        return loss, DummyMSDA()
    elif msda_config['type'] == 'FMix':
        fmix = FMix(loss, decay_power=msda_config['decay_power'], alpha=msda_config['alpha'],
                    log_stats=log_stats, name_prefix=name_prefix)
        fmix_loss = fmix.loss
        return fmix_loss, fmix
    elif msda_config['type'] == 'Mixup':
        mixup = Mixup(loss, alpha=msda_config['alpha'],
                    log_stats=log_stats, name_prefix=name_prefix)
        mixup_loss = mixup.loss
        return mixup_loss, mixup
    else:
        raise NotImplementedError()

