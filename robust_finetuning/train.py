import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable
#from torchvision import datasets, transforms
import torch.optim as optim
import sys, os, inspect
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, '')
from configs import get_config
from utils.Evaluator import Evaluator
from datasets import get_FundusKaggle

import argparse
import time
#from datetime import datetime
import random
import math

from robustness import model_utils, datasets
import robustbench as rb
import robust_finetuning.data as data
from robust_finetuning.autopgd_train import apgd_train
import robust_finetuning.utils_rf as utils
from model_zoo.fast_models import PreActResNet18
import robust_finetuning.other_utils as other_utils
import robust_finetuning.eval as utils_eval

eps_dict = {
    'cifar10': {'Linf': 8. / 255., 'L2': .5, 'L1': 12.},
    'imagenet': {'Linf': 4. / 255., 'L2': 2., 'L1': 255.},
'funduskaggle' : {'Linf': 2. / 255., 'L2': .01, 'L1': 8.}}
    #'funduskaggle' : {'Linf': 2. / 255., 'L2': .15, 'L1': 8.}}
    ##'funduskaggle' : {'Linf': 6. / 255., 'L2': 0.77, 'L1': 103.5}}

here = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Wong2020Fast')
    #parser.add_argument('--eps', type=float, default=8/255)
    #parser.add_argument('--n_ex', type=int, default=100, help='number of examples to evaluate on')
    parser.add_argument('--batch_size_eval', type=int, default=100, help='batch size for evaluation')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--data_dir', type=str, default='/home/scratch/datasets/CIFAR10', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    parser.add_argument('--save_dir', type=str, default='./trained_models')
    #parser.add_argument('--norm', type=str, default='Linf')
    #parser.add_argument('--save_imgs', action='store_true')
    parser.add_argument('--lr-schedule', default='piecewise-ft')
    parser.add_argument('--lr-max', default=.01, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    #parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=-1, help='if -1 no evaluation during training')
    parser.add_argument('--act', type=str, default='softplus1')
    parser.add_argument('--finetune_model', action='store_true')
    parser.add_argument('--l_norms', type=str, default='Linf L1', help='norms to use in adversarial training')
    parser.add_argument('--attack', type=str)
    #parser.add_argument('--pgd_iter', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--l_eps', type=str, help='epsilon values for adversarial training wrt each norm')
    parser.add_argument('--notes_run', type=str, help='appends a comment to the run name')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--l_iters', type=str, help='iterations for each norms in adversarial training (possibly different)')
    #parser.add_argument('--epoch_switch', type=int)
    #parser.add_argument('--save_min', type=int, default=0)
    parser.add_argument('--save_optim', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    #parser.add_argument('--no_wd_bn', action='store_true')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--at_iter', type=int, help='iteration in adversarial training (used for all norms)')
    parser.add_argument('--n_ex_eval', type=int, default=100)
    parser.add_argument('--n_ex_final', type=int, default=100)
    parser.add_argument('--final_eval', action='store_true', help='run long evaluation after training')

    parser.add_argument('--additional_hidden', type=int, default=0)

    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--model_types', type=int, nargs='+', default=None,
                        help='Number of model to use (the last model type '
                             'should be the one for the feature distance, the first - the main model,'
                             'if not FID benchmarking is used)')
    parser.add_argument('--script_type', type=str, default='finetuning', help='Script type for cross project imports')
    parser.add_argument('--data_folder', default='/home/scratch/datasets')
    parser.add_argument('--gpu', '--list', nargs='+', default=[0],
                        help='GPU indices, if more than 1 parallel modules will be called')
    parser.add_argument('--model_epoch_num', type=int, default=None, help='Epoch of the model to load (Currently is being used only for RATIO)')
    parser.add_argument('--project_folder', type=str, default=None, help='folder, where the projected is located')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = get_config(args)
    args.dataset = config.data.dataset.lower()

    if len(args.gpu) == 0:
        args.device = torch.device('cpu')
        args.device_ids = None
        print('Warning! Computing on CPU')
    elif len(args.gpu) == 1:
        args.device_ids = None
        args.device = torch.device('cuda:' + str(args.gpu[0]))
    else:
        args.device_ids = [int(i) for i in args.gpu]
        args.device = torch.device('cuda:' + str(min(args.device_ids)))
        config.sampling.batch_size = config.sampling.batch_size * len(args.device_ids)

    if args.project_folder is None:
        args.project_folder = here

    # logging and saving tools
    utils.get_runname(args)
    print('Logging to', 'runs/'+args.fname)

    writer = SummaryWriter('runs/'+args.fname)

    other_utils.makedir('{}/{}'.format(args.save_dir, args.fname)) #args.save_dir
    args.all_norms = ['L2']  # ['Linf', 'L2', 'L1']
    args.all_epss = [eps_dict[config.data.dataset.lower()][c] for c in args.all_norms]
    stats = utils.stats_dict(args)
    logger = other_utils.Logger('{}/{}/log_train.txt'.format(args.save_dir,
        args.fname))
    log_eval_path = '{}/{}/log_eval_final.txt'.format(args.save_dir, args.fname)
    
    # fix seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    
    # load data
    if config.data.dataset.lower() == 'cifar10':
        train_loader, _ = data.load_cifar10_train(args, only_train=True)

        # non augmented images for statistics
        x_train_eval, y_train_eval = data.load_cifar10(args.n_ex_eval,
            args.data_dir, training_set=True, device='cuda')
        x_test_eval, y_test_eval = data.load_cifar10(args.n_ex_eval,
            args.data_dir, device='cuda') #training_set=True
        
        args.n_cls = 10
    elif config.data.dataset.lower() == 'funduskaggle':
        kwargs = {'split': 'train', 'batch_size': args.batch_size, 'augm_type': 'murat', 'size': 224,
                             'binary': True, 'preprocess': 'hist_eq', 'bg_kernel': 'none',
                             'clahe': True, 'balanced': True, 'data_folder': args.data_folder, 'project_folder': here}

        train_loader = get_FundusKaggle(**kwargs)
        kwargs['postfix'] = '1000_samples'
        kwargs['batch_size'] = args.batch_size_eval
        train_loader_eval = get_FundusKaggle(**kwargs)
        kwargs['postfix'] = None
        kwargs['split'] = 'test'
        kwargs['balanced'] = False
        test_loader = get_FundusKaggle(**kwargs)
        # non augmented images for statistics
        #print('loading train sample')

        #x_train_eval, y_train_eval = data.load_funduskaggle(args.n_ex_eval, d_config=config.data, data_init=True, data_folder=args.data_folder,
        #                                                    project_folder=here, training_set=True, device=args.device)

        #print('loading eval sample')
        #x_test_eval, y_test_eval = data.load_funduskaggle(args.n_ex_eval, d_config=config.data, data_init=True, data_folder=args.data_folder,
        #                                                  project_folder=here, device=args.device)  # training_set=True
        #print('samples loaded')
        args.n_cls = 2

    elif config.data.dataset.lower() == 'imagenet':
        ds = datasets.ImageNet('/home/scratch/datasets/imagenet/')
        train_loader, validation_loader = ds.make_loaders(
            only_val=False, batch_size=args.batch_size, workers=8)

        # non augmented images for statistics
        x_train_eval, y_train_eval = data.load_imagenet_robustness(args.n_ex_eval, training_set=True)
        x_test_eval, y_test_eval = data.load_imagenet_robustness(args.n_ex_eval)  # training_set=True

        args.n_cls = 1000
    else:
        raise NotImplemented
    #print('data loaded on {}'.format(x_test_eval.device))
    
    # load model
    if not args.finetune_model:
        assert config.data.dataset.lower() == 'cifar10'
        #from model_zoo.fast_models import PreActResNet18
        model = PreActResNet18(10, activation=args.act).to(args.device)
        model.eval()
    elif args.model_name.startswith('Max'):
        evaluator = Evaluator(args, config, {})
        assert len(args.model_types) == 1
        print('model type_id is', args.model_types)
        model = evaluator.load_model(
                args.model_types[0],
                return_preloaded_models=False,
                use_temperature=False
            )
        model.to(args.device)
        model.eval()
        #print('state dicts are', model.state_dict(), model.model.state_dict())

    elif args.model_name.startswith('RB'):
        #raise NotImplemented
        model = rb.utils.load_model(args.model_name.split('_')[1], model_dir=args.model_dir,
            dataset=config.data.dataset.lower(), threat_model=args.model_name.split('_')[2])
        model.to(args.device)
        model.eval()
        print('{} ({}) loaded'.format(*args.model_name.split('_')[1:]))
    elif args.model_name.startswith('pretr'):
        model = utils.load_pretrained_models(args.model_name)
        model.to(args.device)
        model.eval()
        print('pretrained model loaded')
    elif args.model_name.startswith('rmtIN1000:'):
        model_arch = args.model_name.split(':')[1]

        print(f'[Replacing the last layer with {args.additional_hidden} '
              f'hidden layers and 1 classification layer that fits the {config.data.dataset.lower()} dataset.]')

        model, checkpoint = model_utils.make_and_restore_model(
            arch=model_arch, dataset=datasets.ImageNet('')) #, resume_path=args.model_dir)

        while hasattr(model, 'model'):
            model = model.model

        model = utils.ft(
            model_arch, model, args.n_cls, args.additional_hidden)

        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=datasets.ImageNet(''),
                                                               add_custom_forward=False)

        model.to(args.device)
        model.eval()


    # set loss
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # set optimizer
    if args.weight_decay > 0 and not args.finetune_model: #args.no_wd_bn
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay': args.weight_decay},
                  {'params': no_decay, 'weight_decay': 0}]
        print('not using wd for bn layers')
    else:
        params = model.parameters()
    optimizer = optim.SGD(params, lr=1., momentum=0.9,
        weight_decay=args.weight_decay)

    # get lr scheduler
    lr_schedule = utils.get_lr_schedule(args)

    if args.attack is not None:
        # set norms, eps and iters for training
        args.l_norms = args.l_norms.split(' ')
        if args.l_eps is None:
            args.l_eps = [eps_dict[config.data.dataset.lower()][c] for c in args.l_norms]
        else:
            args.l_eps = [float(c) for c in args.l_eps.split(' ')]
        if args.l_iters is not None:
            args.l_iters = [int(c) for c in args.l_iters.split(' ')]
        else:
            args.l_iters = [args.at_iter + 0 for _ in args.l_norms]
        print('[train] ' + ', '.join(['{} eps={:.5f} iters={}'.format(
            args.l_norms[c], args.l_eps[c], args.l_iters[c]) for c in range(len(
            args.l_norms))]))

        # set eps for evaluation
        print('all epss before', args.all_epss)
        for i, norm in enumerate(args.l_norms):
            idx = args.all_norms.index(norm)
            args.all_epss[idx] = args.l_eps[i] + 0.
        print('[eval] ' + ', '.join(['{} eps={:.5f}'.format(args.all_norms[c],
            args.all_epss[c]) for c in range(len(args.all_norms))]))
        print('all epss after', args.all_epss)

    acc_test = utils_eval.eval_norms_fast_loader(model, test_loader,
                                                 args.all_norms, args.all_epss, n_iter=10, n_cls=args.n_cls,
                                                 bs=args.batch_size_eval)

    keys = args.all_norms + ['union', 'clean']
    keysBalanced = [key + 'Balanced' for key in keys]
    all_keys = keys + keysBalanced
    str_test, str_train = '', ''
    for norm in all_keys:
        stats['rob_acc_test_dets'][norm][0] = acc_test[norm]
        writer.add_scalar('Test/' + norm, acc_test[norm] * 100, 0)

        str_test += ' {} {:.1%}'.format(norm, acc_test[norm])

    print('[eval test]{}'.format(str_test))

    # training loop
    for epoch in tqdm(range(0, args.epochs), desc='Epochs'):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        running_acc = 0.
        running_acc_ep = 0.
        startt = time.time()
        if epoch == 0: #epoch_init
            acc_norms = [[0., 0.] for _ in range(len(args.l_norms))]
        loss_norms = {k: [0., 0.] for k in args.l_norms}

        time_prev = time.time()
        for i, out in tqdm(enumerate(train_loader), desc='Batches'):
            if len(out) == 2:
                x_loader, y_loader = out
            elif len(out) == 3:
                x_loader, y_loader, _ = out
            else:
                raise ValueError('Unexpected number of elements in dataloader.')
            x, y = x_loader.to(args.device), y_loader.to(args.device)

            # update lr
            lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            writer.add_scalar('Train/LR', lr, epoch + 1)
            optimizer.param_groups[0].update(lr=lr)

            if args.attack is not None:
                model.eval()
                
                # sample which norm to use for the current batch
                if all([val[1] > 0 for val in acc_norms]):
                    ps = [val[0] / val[1] for val in acc_norms]
                else:
                    ps = [.5] * len(acc_norms)
                ps = [1. - val for val in ps]
                norm_curr = random.choices(range(len(ps)), weights=ps)[0]

                # compute training points
                if args.attack == 'apgd':
                    x_tr, acc_tr, _, _ = apgd_train(model, x, y, norm=args.l_norms[norm_curr],
                        eps=args.l_eps[norm_curr], n_iter=args.l_iters[norm_curr])
                    y_tr = y.clone()
                else:
                    raise NotImplemented
                
                # update statistics
                acc_norms[norm_curr][0] += acc_tr.sum()
                acc_norms[norm_curr][1] += x.shape[0]
                
                model.train()
            else:
                # standard training
                x_tr = x.clone()
                y_tr = y.clone()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if args.loss in ['ce']:
                outputs = model(x_tr)
                loss = criterion(outputs, y_tr)
            loss.backward()
            optimizer.step()

            # collect stats
            running_loss += loss.item() #w_tr
            #running_acc += (outputs.max(dim=-1)[1] == y_tr).cpu().float().sum().item()
            running_acc_ep += (outputs.max(dim=-1)[1] == y_tr).cpu().float().sum().item()
            
            # track loss for each norm
            if not args.attack is None:
                loss_norms[args.l_norms[norm_curr]][0] += loss.item()
                loss_norms[args.l_norms[norm_curr]][1] += 1

            # logging
            time_iter = time.time() - time_prev
            time_prev = time.time()
            time_cum = time.time() - startt
            if len(args.l_norms) > 0:
                other_stats = ' [indiv] ' + ', '.join(['{} {:.5f}'.format(k,
                    v[0] / max(1, v[1])) for k, v in loss_norms.items()])
            else:
                other_stats = ''
            print('batch {} / {} [time] iter {:.1f} s, cum {:.1f} s, exp {:.1f} s [loss] {:.5f} [acc] {:.1%}{}'.format(
                i + 1, len(train_loader), time_iter, time_cum,
                time_cum / (i + 1) * len(train_loader), running_loss / (i + 1),
                running_acc_ep / (i + 1) / args.batch_size, other_stats), end='\r')

        model.eval()
        
        # training stats
        stats['loss_train_dets']['clean'][epoch] = running_loss / len(train_loader)
        if not args.attack is None:
            for norm_curr in args.l_norms:
                stats['loss_train_dets'][norm_curr][epoch] = loss_norms[norm_curr][0
                    ] / loss_norms[norm_curr][1]
                stats['freq_in_at'][norm_curr][epoch] = loss_norms[norm_curr][1
                    ] / len(train_loader)
        str_to_log = '[epoch] {} [time] {:.1f} s [train] loss {:.5f}'.format(
                epoch + 1, time.time() - startt, stats['loss_train_dets']['clean'][epoch]) #stats['rob_acc_train_dets']['clean'][epoch]
        
        # compute robustness stats (apgd with 100 iterations)
        if (epoch + 1) % args.eval_freq == 0 and args.eval_freq > -1:
            # training points
            acc_train = utils_eval.eval_norms_fast_loader(model, train_loader_eval,
                args.all_norms, args.all_epss, n_iter=10, n_cls=args.n_cls, bs=args.batch_size_eval)
            # test points
            acc_test = utils_eval.eval_norms_fast_loader(model, test_loader,
                args.all_norms, args.all_epss, n_iter=10, n_cls=args.n_cls, bs=args.batch_size_eval)
            str_test, str_train = '', ''
            keys = args.all_norms + ['union', 'clean']
            keysBalanced = [key + 'Balanced' for key in keys]
            print(keys)
            print(keysBalanced)
            print(all_keys)
            for norm in all_keys:
                stats['rob_acc_test_dets'][norm][epoch+1] = acc_test[norm]
                writer.add_scalar('Test/' + norm, acc_test[norm]*100, epoch + 1,)
                stats['rob_acc_train_dets'][norm][epoch+1] = acc_train[norm]
                writer.add_scalar('Train/' + norm, acc_train[norm]*100, epoch + 1)

                str_test += ' {} {:.1%}'.format(norm, acc_test[norm])
                str_train += ' {} {:.1%}'.format(norm, acc_train[norm])

            str_to_log += '\n'
            str_to_log += '[eval train]{} [eval test]{}'.format(str_train, str_test)
        
        # saving
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            curr_dict = model.model.state_dict()
            if args.save_optim:
                curr_dict = {'state_dict': model.model.state_dict(), 'optim': optimizer.state_dict()}
            torch.save(curr_dict, '{}/{}/ep_{}.pth'.format(
                    args.save_dir, args.fname, epoch + 1))
            torch.save(stats, '{}/{}/metrics.pth'.format(args.save_dir, args.fname))

        logger.log(str_to_log)

    # run long eval
    if args.final_eval:
        if config.data.dataset.lower() == 'cifar10':
            x, y = data.load_cifar10(args.n_ex_final, data_dir=args.data_dir, device='cpu')
        elif config.data.dataset.lower() == 'funduskaggle':
            x, y = data.load_funduskaggle(n_examples=args.n_ex_final, d_config=config.data, data_init=True,
                                                         data_folder=args.data_folder,
                                                         project_folder=args.project_folder, device='cpu')
            #x, y = data.load_funduskaggle(args.n_ex_final, device='cpu')

        l_x_adv, stats['final_acc_dets'] = utils_eval.eval_norms(model, x, y,
            l_norms=args.all_norms, l_epss=args.all_epss,
            bs=args.batch_size_eval, log_path=log_eval_path) #model, args=args
        for key, val in dict(stats['final_acc_dets']).items():
            writer.add_scalar('Test/'+key, val*100, epoch + 1)
        torch.save(stats, '{}/{}/metrics.pth'.format(args.save_dir, args.fname))
        for norm, eps, v in zip(args.l_norms, args.l_eps, l_x_adv):
            torch.save(v,  '{}/{}/eval_{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, args.fname, 'final', norm, args.n_ex_final, eps))


if __name__ == '__main__':
    main()

