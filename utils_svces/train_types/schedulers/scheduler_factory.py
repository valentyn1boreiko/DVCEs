""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .step_lr import StepLRScheduler

def create_scheduler(args, optimizer):
    num_epochs = args['cycle_length']
    noise_range = None

    if args['scheduler_type'] == 'CosineAnnealing':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=args['lr_min'],
            warmup_t=args['warmup_length'],
            t_mul=args['cycle_multiplier'],
            noise_range_t=noise_range,
        )
    elif args['scheduler_type'] == 'StepLR':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_epochs=args['decay_epochs'],
            decay_rate=args['decay_rate'],
            warmup_t=args['warmup_length'],
            noise_range_t=noise_range,
        )
    else:
        raise NotImplementedError(f'Scheduler {args.sched} not implemented')

    return lr_scheduler, num_epochs