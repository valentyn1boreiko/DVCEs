import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from datetime import datetime

from other_utils import makedir


models = os.listdir('./trained_models')
models_names = [c for c in models if '05-25' in c]

l_skip = ['rob_acc_train', 'rob_acc_test', 'clean_acc_test', 'final_acc_dets']

makedir('./plots_train')
logpath = './results_train/stats_{}.txt'.format(str(datetime.now())[:10])
l_runs = []

for c in models_names:
    if os.path.exists('./trained_models/{}/metrics.pth'.format(c)):
        metrics = torch.load('./trained_models/{}/metrics.pth'.format(c))
        n_plt = len([c for c in metrics if not c in l_skip])
        fig = plt.figure(figsize=(5 * n_plt, 4))
        ax = []
        i_plt = 0
        for i, (k, v) in enumerate(metrics.items()):
            if not k in l_skip:
                ax.append( fig.add_subplot(1, n_plt, i_plt + 1))
                if not isinstance(v, dict):
                    plt.plot(v.numpy())
                else:
                    leg = []
                    for a, b in v.items():
                        plt.plot(b.numpy())
                        leg.append(a)
                    ax[-1].legend(leg)
                ax[-1].set_title(k)
                ax[-1].grid()
                i_plt += 1
        
        if 'final_acc_dets' in metrics.keys():
            acc_dets = '\n\n' + ', '.join(['{} {:.1%}'.format(k, v) for k, v in metrics[
                'final_acc_dets']])
            l_runs.append('{} - {}'.format(c, acc_dets.split('\n')[2]))
        else:
            acc_dets = ''
        fig.suptitle(c + acc_dets, y=1.1)
        plt.savefig('./trained_models/{}/pl_stats.png'.format(c), bbox_inches='tight')
        plt.savefig('./plots_train/{}.pdf'.format(c), bbox_inches='tight')
        plt.close()

'''
with open(logpath, 'w') as f:
    f.write('\n'.join(l_runs))
    f.flush()
'''

