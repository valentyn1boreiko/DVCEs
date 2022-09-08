def create_fmix_config(decay_power=3.0, alpha=1.0):
    fmix_config = {'type': 'FMix', 'decay_power': decay_power, 'alpha': alpha}
    return fmix_config

def create_mixup_config(alpha=1.0):
    fmix_config = {'type': 'Mixup', 'alpha': alpha}
    return fmix_config
