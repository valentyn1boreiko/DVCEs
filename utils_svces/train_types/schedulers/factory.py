# import torch.optim as optim
# import math
# import numpy as np
#
# def get_scheduler(optimizer, lr_scheduler_config):
#     if lr_scheduler_config['scheduler_type'] == 'StepLR':
#         batchwise_scheduler = False
#         scheduler = optim.lr_scheduler.StepLR(optimizer, lr_scheduler_config['step_size'],
#                                                    lr_scheduler_config['gamma'],
#                                                    lr_scheduler_config['last_epoch'])
#     elif lr_scheduler_config['scheduler_type'] == 'ExponentialLR':
#         batchwise_scheduler = False
#         scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_scheduler_config['gamma'],
#                                                           lr_scheduler_config['last_epoch'])
#     elif lr_scheduler_config['scheduler_type'] == 'PiecewiseConstant':
#         batchwise_scheduler = False
#
#         def piecewise(epoch):
#             for stage_end, stage_factor in zip(lr_scheduler_config['epoch_stages'],
#                                                lr_scheduler_config['stages_factors']):
#                 if epoch < stage_end:
#                     return stage_factor
#             print(f'Warning: Epoch {epoch} not in epoch stages')
#             return lr_scheduler_config['stages_factors'][-1]
#
#         scheduler = optim.lr_scheduler.LambdaLR(optimizer, piecewise)
#     elif lr_scheduler_config['scheduler_type'] == 'CosineAnnealing':
#         batchwise_scheduler = True
#         period_length_batches = lr_scheduler_config['period_length_batches']
#
#         def cosine_annealing(step, period_length_batches, lr_min, lr_max):
#             return lr_min + (lr_max - lr_min) * 0.5 * (
#                     1 + math.cos((step % period_length_batches) / period_length_batches * math.pi))
#
#         cosine_lambda = lambda x: (lr_scheduler_config['period_falloff'] ** np.floor(
#             x / period_length_batches)) * cosine_annealing(x, period_length_batches, lr_scheduler_config['lr_min'],
#                                                            lr_scheduler_config['lr_max'])
#
#         if 'warmup_length_batches' in lr_scheduler_config:
#             warmup_length = lr_scheduler_config['warmup_length_batches']
#             lr_lambda = lambda x: min(cosine_lambda(x), lr_scheduler_config['lr_max'] * x / warmup_length )
#         else:
#             lr_lambda = cosine_lambda
#
#         scheduler = optim.lr_scheduler.LambdaLR(
#             optimizer, lr_lambda=lr_lambda)
#     elif lr_scheduler_config['scheduler_type'] == 'CyclicalLR':
#         batchwise_scheduler = True
#         # Scaler: we can adapt this if we do not want the triangular CLR
#         period_length_batches = lr_scheduler_config['period_length_batches']
#         midpoint = lr_scheduler_config['midpoint'] * period_length_batches
#         period_falloff = lr_scheduler_config['period_falloff']
#         xp = np.array([0, midpoint, period_length_batches])
#         yp = np.array([lr_scheduler_config['lr_start'], lr_scheduler_config['lr_mid'],
#                        lr_scheduler_config['lr_end']])
#
#         def cylic_lr(x):
#             period_factor = (period_falloff ** np.floor(x / period_length_batches))
#             interp = np.interp(x % period_length_batches, xp, yp)
#             return period_factor * interp
#
#         scheduler = optim.lr_scheduler.LambdaLR(optimizer, [cylic_lr])
#     elif lr_scheduler_config['scheduler_type'] == 'LogarithmicFindLRScheduler':
#         batchwise_scheduler = True
#         q = math.pow(lr_scheduler_config['lr_end'] / lr_scheduler_config['lr_start'],
#                      1 / (lr_scheduler_config['period_length_batches'] - 1))
#
#         def log_scheduler(step, lr_start, q):
#             return lr_start * q ** step
#
#         lr_lambda = lambda x: log_scheduler(x, lr_scheduler_config['lr_start'], q)
#         scheduler = optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda])
#     else:
#         raise ValueError('Scheduler not supported {}'.format(lr_scheduler_config['scheduler_type']))
#
#     return scheduler, batchwise_scheduler
