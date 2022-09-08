import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys

import robustbench as rb
import data
#from autopgd_train import apgd_train
from robust_finetuning import utils_rf
from robust_finetuning.model_zoo.fast_models import PreActResNet18
from robust_finetuning import other_utils
from auto_attack import autoattack
from robust_finetuning.autopgd_train import apgd_train

eps_dict = {'cifar10': {'Linf': 8. / 255., 'L2': .5, 'L1': 12.},
    'imagenet': {'Linf': 4. / 255., 'L2': 2., 'L1': 255.}}

def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   n_cls = 10,
                   batch_size: int = 100,
                   device: torch.device = None,
                   multiple_models=False):
    if device is None:
        device = x.device
    acc = 0.
    correct_per_class = torch.zeros((n_cls,), dtype=x.dtype, device='cpu')
    all_per_class = torch.zeros((n_cls,), dtype=x.dtype, device='cpu')

    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            if multiple_models:
                print('using uniform weights, eval')
                output1 = model[0](x_curr)
                output2 = model[1](x_curr)
                output = output1.softmax(1)*0.5 + output2.softmax(1)*0.5
                acc += (output.max(1)[1] == y_curr).float().sum()
            else:
                output = model(x_curr)
                acc += (output.max(1)[1] == y_curr).float().sum()
            correct_per_class.index_add_(0, y_curr.to('cpu'), (output.max(1)[1] == y_curr).type(x_curr.dtype).to('cpu'))
            all_per_class.index_add_(0, y_curr.to('cpu'), torch.ones((x_curr.shape[0], ), dtype=x_curr.dtype).to('cpu'))

    print('accuracies', acc.item() / x.shape[0], (correct_per_class.sum() / all_per_class.sum()).item())
    print('correct_per_class', correct_per_class)
    print('all_per_class', all_per_class)

    #assert acc.item() / x.shape[0] == (correct_per_class.sum() / all_per_class.sum()).item()
    bal_acc = (correct_per_class / all_per_class).sum().item() / n_cls
    return acc.item() / x.shape[0], bal_acc


def eval_single_norm(model, x, y, norm='Linf', eps=8. / 255., bs=1000,
    log_path=None, verbose=True, multiple_models=False, n_cls=2, masks=1):
    if multiple_models:
        adversary = autoattack.AutoAttack(model[0], norm=norm, eps=eps,
            log_path=log_path, use_fw=norm not in ['L2', 'L1', 'Linf'], version='binary', second_classifier=model[1], masks=masks) # remove version binary for cifar10, IN attacks
    else:
        adversary = autoattack.AutoAttack(model, norm=norm, eps=eps,
                                          log_path=log_path, use_fw=norm not in ['L2', 'L1', 'Linf'], version='binary',
                                          second_classifier=None)  # remove version binary for cifar10, IN attacks
    ##adversary.attacks_to_run = ['apgd-ce', 'apgd-t'] # uncomment it for cifar10, IN attacks
    #adversary.attacks_to_run = ['apgd-t']
    #adversary.apgd.n_restarts = 1
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x, y, bs=bs)
    #if verbose
    acc, bal_acc = clean_accuracy(model, x_adv, y, device='cuda', n_cls=n_cls, multiple_models=multiple_models)
    other_utils.check_imgs(x_adv, x, norm)
    print('robust accuracy: {:.1%}'.format(acc))
    return x_adv


def eval_norms(model, x, y, l_norms, l_epss, bs=1000, log_path=None, n_cls=10, multiple_models=False, masks=1):
    print('num classes is', n_cls)
    l_x_adv = []
    acc_dets = []
    logger = other_utils.Logger(log_path)

    predBalanced = torch.zeros((n_cls,), dtype=x.dtype)
    all_per_class = torch.zeros((n_cls,), dtype=x.dtype)

    print('all norms', list(zip(l_norms, l_epss)))
    for norm, eps in zip(l_norms, l_epss):
        print('norm eps', norm, eps)
        x_adv_curr = eval_single_norm(model, x, y, norm=norm, eps=eps, bs=bs,
            log_path=log_path, verbose=False, multiple_models=multiple_models, n_cls=n_cls, masks=masks)
        l_x_adv.append(x_adv_curr.cpu())
    acc, bal_acc, output, total, total_balanced = utils_rf.get_accuracy_and_logits(model, x, y, batch_size=bs,
                                                            n_classes=n_cls, multiple_models=multiple_models)
    acc = acc / total
    acc_check = bal_acc.sum() / total_balanced.sum()
    bal_acc = (bal_acc / total_balanced).sum() / n_cls
    pred = output.to(y.device).max(1)[1] == y
    logger.log('')
    logger.log('clean accuracy: {:.1%}'.format(pred.float().mean()))
    logger.log('clean accuracy, check balanced: {:.1%}'.format(acc_check.float()))

    print('clean accuracy: {:.1%}'.format(acc))
    acc_dets.append(('clean', acc + 0.))
    acc_dets.append(('cleanBalanced', bal_acc + 0.))
    for norm, eps, x_adv in zip(l_norms, l_epss, l_x_adv):
        acc, bal_acc, output, total, total_balanced = utils_rf.get_accuracy_and_logits(model, x_adv, y,
                                                                batch_size=bs, n_classes=n_cls, multiple_models=multiple_models)
        acc = acc / total
        acc_check = bal_acc.sum() / total_balanced.sum()
        bal_acc = (bal_acc / total_balanced).sum() / n_cls

        other_utils.check_imgs(x_adv, x.cpu(), norm)
        pred_curr = output.to(y.device).max(1)[1] == y
        logger.log('robust accuracy, balanced accuracy {}: {:.1%}, {:.1%}'.format(norm, pred_curr.float().mean(), bal_acc))
        logger.log('check robust accuracy, check acc. {} : {:.1%}, {:.1%}'.format(norm, acc, acc_check))
        pred *= pred_curr
        acc_dets.append((norm, acc + 0.))
        acc_dets.append((norm+'Balanced', bal_acc + 0.))
    logger.log('robust accuracy {}: {:.1%}'.format('+'.join(l_norms),
        pred.float().mean()))

    predBalanced.index_add_(0, y, pred.type(x.dtype))
    all_per_class.index_add_(0, y, torch.ones((x.shape[0], ), dtype=x.dtype))

    acc_dets.append(('union', pred.float().mean()))
    acc_dets.append(('unionBalanced', (predBalanced / all_per_class).sum() / n_cls))
    return l_x_adv, acc_dets

def eval_norms_fast_loader(model, loader, l_norms, l_epss, n_iter=100, n_cls=10, bs=100):
    acc_dict = {}
    bal_acc_dict = {}
    assert not model.training

    for norm in l_norms:
        acc_dict[norm] = 0.
        bal_acc_dict[norm] = 0.

    acc_dict['clean'] = 0.
    bal_acc_dict['clean'] = 0.

    acc_dict['total'] = 0.
    bal_acc_dict['total'] = 0.
    predBalanced = torch.zeros((n_cls,)).cuda()
    pred_all = 0

    for i, out in enumerate(loader):
        if len(out) == 3:
            x, y, _ = out
        elif len(out) == 2:
            x, y = out
        acc, bal_acc, output, total, total_bal = utils_rf.get_accuracy_and_logits(model, x.cuda(), y.cuda(), batch_size=bs,
                                                                   n_classes=n_cls)
        assert total == x.shape[0]

        pred = output.to(y.device).max(1)[1] == y
        all_per_class = torch.zeros((n_cls,), dtype=x.dtype).cuda()

        acc_dict['clean'] += acc
        bal_acc_dict['clean'] += bal_acc
        acc_dict['total'] += total
        bal_acc_dict['total'] += total_bal

        assert acc_dict['total'] == bal_acc_dict['total'].sum()

        for norm, eps in zip(l_norms, l_epss):
            _, _, _, x_adv = apgd_train(model, x.cuda(), y.cuda(), norm=norm, eps=eps,
                                        n_iter=n_iter, is_train=False)
            acc_norm_temp, bal_acc_norm_temp, output, _, _ = utils_rf.get_accuracy_and_logits(model,
                                                                                            x_adv, y.cuda(), batch_size=bs,
                                                                                          n_classes=n_cls)
            acc_dict[norm] += acc_norm_temp
            bal_acc_dict[norm] += bal_acc_norm_temp

            pred *= output.to(y.device).max(1)[1] == y

        if i == 0:
            predBalanced = predBalanced.type(x.dtype)
        predBalanced.index_add_(0, y.cuda(), pred.type(x.dtype).cuda())
        pred_all += pred.sum()
    acc_dict['union'] = pred_all.float() / acc_dict['total']
    acc_dict['unionBalanced'] = (predBalanced / bal_acc_dict['total']).sum().item() / n_cls

    for norm in l_norms + ['clean']:
        acc_dict[norm] = acc_dict[norm] / acc_dict['total']
        acc_dict[norm+'Balanced'] = (bal_acc_dict[norm] / bal_acc_dict['total']).sum().item() / n_cls

    return acc_dict

def eval_norms_fast(model, x, y, l_norms, l_epss, n_iter=100, n_cls=10):
    acc_dict = {}
    bal_acc_dict = {}
    assert not model.training
    bs = x.shape[0]
    acc, bal_acc, output = utils_rf.get_accuracy_and_logits(model, x, y, batch_size=bs,
                                                            n_classes=n_cls)
    pred = output.to(y.device).max(1)[1] == y
    predBalanced = torch.zeros((n_cls, ), dtype=x.dtype).cuda()
    all_per_class = torch.zeros((n_cls, ), dtype=x.dtype).cuda()

    acc_dict['clean'] = acc + 0.
    bal_acc_dict['clean'] = acc + 0.
    for norm, eps in zip(l_norms, l_epss):
        _, _, _, x_adv = apgd_train(model, x, y, norm=norm, eps=eps,
            n_iter=n_iter, is_train=False)
        acc_dict[norm], bal_acc_dict[norm], output = utils_rf.get_accuracy_and_logits(model,
                                                                                      x_adv, y, batch_size=bs, n_classes=n_cls)
        pred *= output.to(y.device).max(1)[1] == y

    predBalanced.index_add_(0, y.cuda(), pred.type(x.dtype).cuda())
    all_per_class.index_add_(0, y.cuda(), torch.ones((x.shape[0], ), dtype=x.dtype).cuda())
    acc_dict['union'] = pred.float().mean()
    bal_acc_dict['union'] = (predBalanced / all_per_class).sum().item() / n_cls
    return acc_dict, bal_acc_dict

"""
def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   n_cls = 10,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    correct_per_class = torch.zeros((n_cls,), dtype=x.dtype, device='cpu')
    all_per_class = torch.zeros((n_cls,), dtype=x.dtype, device='cpu')

    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
            correct_per_class.index_add_(0, y_curr.to('cpu'), (output.max(1)[1] == y_curr).type(x.dtype).to('cpu'))
            all_per_class.index_add_(0, y_curr.to('cpu'), torch.ones((x.shape[0], ), dtype=x.dtype).to('cpu'))

    print('accuracies', acc.item() / x.shape[0], (correct_per_class.sum() / all_per_class.sum()).item())
    print('correct_per_class', correct_per_class)
    print('all_per_class', all_per_class)

    assert acc.item() / x.shape[0] == (correct_per_class.sum() / all_per_class.sum()).item()
    bal_acc = (correct_per_class / all_per_class).sum().item() / n_cls
    return acc.item() / x.shape[0], bal_acc
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Wong2020Fast')
    parser.add_argument('--n_ex', type=int, default=100, help='number of examples to evaluate on')
    parser.add_argument('--batch_size_eval', type=int, default=100, help='batch size for evaluation')
    #parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--data_dir', type=str, default='/home/scratch/datasets/CIFAR10', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    parser.add_argument('--save_dir', type=str, default='./trained_models')
    parser.add_argument('--l_norms', type=str, default='Linf L2 L1')
    parser.add_argument('--l_eps', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--only_clean', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    x_test, y_test = data.load_cifar10(1000, data_dir='/home/scratch/datasets/CIFAR10')

    model = utils.load_pretrained_models('pretr_L2') #args.model_name
    ckpt = torch.load('./trained_models/model_2021-04-21 19:57:33.710832 cifar10 lr=0.05000 piecewise-ft ep=3 attack=apgd fts=pretr_L1 seed=0 at=Linf L1 eps=default iter=10/ep_3.pth')
    model.load_state_dict(ckpt)
    model.eval()
    
    #eval_single_norm(model, x_test, y_test, norm='L2', eps=.5, bs=256)
    eval_norms(model, x_test, y_test, l_norms=['L2', 'L1'], l_epss=[.5, 12.], bs=256)
    '''

    args = parse_args()

    # load data
    if args.dataset == 'cifar10':
        x_test, y_test = data.load_cifar10(args.n_ex, data_dir=args.data_dir,
            device='cpu')
        #x_test, y_test = x_test.cpu(), y_test.cpu()
    
    if os.path.isfile(args.model_name):
        pretr_model = args.model_name.split('fts=')[1].split(' ')[0]
        args.save_dir, ckpt_name = os.path.split(args.model_name) #os.path.join(args.model_name.split('/')[:-1])
        ckpt = torch.load(args.model_name)
    else:
        pretr_model = args.model_name
        args.save_dir = '{}/{}'.format(args.save_dir, args.model_name)
        other_utils.makedir(args.save_dir)
        ckpt_name = 'pretrained'
    not_pretr = os.path.isfile(args.model_name)
    log_path = '{}/log_eval_{}.txt'.format(args.save_dir, ckpt_name)
    
    # load model
    if pretr_model == 'rand':
        model = PreActResNet18(10, activation=args.act).cuda()
        #model.eval()
    elif pretr_model.startswith('RB'):
        model = rb.utils.load_model(pretr_model.split('_')[1], model_dir=args.model_dir,
            dataset=args.dataset, threat_model=pretr_model.split('_')[2])
        model.cuda()
        #model.eval()
        print('{} ({}) loaded'.format(*pretr_model.split('_')[1:]))
    elif pretr_model.startswith('pretr'):
        model = utils_rf.load_pretrained_models(pretr_model)
        print('pretrained model loaded')
    if not_pretr:
        model.load_state_dict(ckpt)
    model.eval()

    # clean acc
    acc = rb.utils.clean_accuracy(model, x_test, y_test,
        device='cuda')
    print('clean accuracy {:.1%}'.format(acc))
    if args.only_clean:
        sys.exit()
    
    
    # set norms and eps
    args.l_norms = args.l_norms.split(' ')
    if args.l_eps is None:
        args.l_eps = [eps_dict[args.dataset][c] for c in args.l_norms]
    else:
        args.l_eps = [float(c) for c in args.l_eps.split(' ')]

    # run attacks
    l_x_adv, _ = eval_norms(model, x_test, y_test, l_norms=args.l_norms,
        l_epss=args.l_eps, bs=args.batch_size_eval, log_path=log_path)

    # saving
    for norm, eps, v in zip(args.l_norms, args.l_eps, l_x_adv):
        torch.save(v,  '{}/eval_{}_{}_1_{}_eps_{:.5f}.pth'.format(
            args.save_dir, ckpt_name, norm, args.n_ex, eps))




