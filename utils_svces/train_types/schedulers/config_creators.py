
def create_cosine_annealing_scheduler_config(cycle_length, lr_min, cycle_multiplier=1, warmup_length=0):
    scheduler_config = {'cycle_length': cycle_length, 'scheduler_type': 'CosineAnnealing',
                        'lr_min': lr_min, 'cycle_multiplier': cycle_multiplier,
                        'warmup_length': warmup_length}
    return scheduler_config

def create_piecewise_consant_scheduler_config(epochs, decay_epochs, decay_rate, warmup_length=0):
    scheduler_config = {'cycle_length': epochs, 'scheduler_type': 'StepLR',
                        'decay_epochs': decay_epochs, 'decay_rate': decay_rate,
                        'warmup_length': warmup_length}
    return scheduler_config

