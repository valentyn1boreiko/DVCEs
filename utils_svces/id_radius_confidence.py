import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pathlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import utils_svces.adversarial_attacks as aa
from PIL import Image
from utils_svces.load_trained_model import load_model
from utils_svces.train_types.train_loss import MaxConfidenceLoss, NegativeWrapper
from auto_attack.autopgd_pt import APGDAttack_singlestepsize as APGDAttack

def id_radius_confidence(model_descriptions, radii, plot_radii, dataloader, bs, datapoints, class_labels, device, eval_dir, dataset, device_ids=None):
    #[,,0] = gt conf
    #[,,1]= other conf
    model_radii_confs = torch.zeros(len(model_descriptions), len(radii), 2)

    num_batches = int(np.ceil(datapoints / bs))

    data_iterator = iter(dataloader)

    data_batches = []
    for _ in range(num_batches):
        data, target = next(data_iterator)
        data_batches.append((data, target))


    for model_idx, (type, folder, checkpoint, temperature, temp) in enumerate(model_descriptions):
        dir = f'{eval_dir}/{folder}_{checkpoint}/IDWorstCase/'
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        model = load_model(type, folder, checkpoint, temperature, device, load_temp=temp, dataset=dataset)
        if device_ids is not None and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()

        imgs = torch.zeros((len(radii), datapoints, 3, 32, 32), dtype=torch.float32)
        targets = torch.zeros(datapoints, dtype=torch.long)
        gt_confidences = torch.zeros((len(radii), datapoints), dtype=torch.float32)
        others_confidences = torch.zeros((len(radii), datapoints), dtype=torch.float32)
        others_predictions = torch.zeros((len(radii), datapoints), dtype=torch.long)

        datapoint_idx = 0
        for batch_idx, (data, target) in enumerate(data_batches):
            data = data.to(device)
            target = target.to(device)
            for radius_idx, radius in enumerate(radii):
                if radius > 1e-8:
                    step_multiplier = 5
                    att = APGDAttack(model, n_restarts=1, n_iter=100 * step_multiplier, n_iter_2=22 * step_multiplier,
                                     n_iter_min=6 * step_multiplier, size_decr=3,
                                     eps=radius, show_loss=False, norm='L2', loss='diff_logit_target', eot_iter=1,
                                     thr_decr=.75, seed=0, normalize_logits=True,
                                     show_acc=False)

                    # adv_samples = att(ref_data, None)
                    adv_samples = att.perturb(data, target)[1]
                else:
                    adv_samples = data

                imgs[radius_idx, datapoint_idx:(datapoint_idx + data.shape[0]), :] = adv_samples.detach().cpu()
                targets[datapoint_idx:(datapoint_idx + data.shape[0])] = target.detach().cpu()

                confidences = F.softmax(model(adv_samples), dim=1)
                gt_confidence = confidences[range(data.shape[0]), target]

                gt_confidences[radius_idx, datapoint_idx:(datapoint_idx + data.shape[0])] = gt_confidence.detach().cpu()

                confidences[range(data.shape[0]), target] = -1e13
                other_conf, other_pred = torch.max(confidences, dim=1)

                others_confidences[radius_idx, datapoint_idx:(datapoint_idx + data.shape[0])] = other_conf.detach().cpu()
                others_predictions[radius_idx, datapoint_idx:(datapoint_idx + data.shape[0])] = other_pred

            datapoint_idx += adv_samples.shape[0]

        model_radii_confs[model_idx, :, 0] = torch.mean(gt_confidences, dim=1)
        model_radii_confs[model_idx, :, 1] = torch.mean(others_confidences, dim=1)

        print(f'{folder}')
        print(f'GT: {model_radii_confs[model_idx, :, 0]}')
        print(f'Other: {model_radii_confs[model_idx, :, 1]}')

        num_radii_to_plot = 0
        for plot_radius in plot_radii:
            if plot_radius:
                num_radii_to_plot += 1

        for img_idx in range(datapoints):
            scale_factor = 1.5
            fig, axs = plt.subplots(1, num_radii_to_plot,
                                    figsize=(scale_factor * num_radii_to_plot, 1.3 * scale_factor))

            col_idx = 0
            for radius_idx in range(len(radii)):
                if plot_radii[radius_idx]:
                    axs[col_idx].axis('off')
                    axs[col_idx].title.set_text(
                        f'{class_labels[targets[img_idx]]} - {gt_confidences[radius_idx, img_idx]:.2f}\n{class_labels[others_predictions[radius_idx, img_idx]]} - {others_confidences[radius_idx, img_idx]:.2f}')
                    img_cpu = imgs[radius_idx, img_idx, :].permute(1, 2, 0)
                    axs[col_idx].imshow(img_cpu, interpolation='lanczos')
                    col_idx += 1

            plt.tight_layout()
            fig.savefig(f'{dir}img_{img_idx}.png')
            fig.savefig(f'{dir}img_{img_idx}.pdf')
            plt.close(fig)

        #animated gif parts
        rows = int(np.sqrt(bs))
        cols = int(np.ceil(bs / rows))
        scale_factor = 4

        for radius_idx, radius in enumerate(radii):
            fig, axs = plt.subplots(rows, cols, figsize=(scale_factor * cols, scale_factor * rows))
            for img_idx in range(data.shape[0]):
                row_idx = int(img_idx / cols)
                col_idx = int(img_idx % cols)

                img_cpu = imgs[radius_idx,img_idx,:].permute(1,2,0)

                axs[row_idx, col_idx].axis('off')
                axs[row_idx, col_idx].imshow(img_cpu)

            fig.savefig(f'{dir}img_gif_part_{radius_idx}.png')
            plt.close(fig)

    torch.save(model_radii_confs, f'{eval_dir}ID_model_radii_confs.pt')
    return model_radii_confs