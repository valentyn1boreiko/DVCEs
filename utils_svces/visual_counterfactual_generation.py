import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import os
import pathlib
import matplotlib as mpl

from blended_diffusion.utils_blended.model_normalization import ResizeWrapper

mpl.use('Agg')
import matplotlib.pyplot as plt
from utils_svces.load_trained_model import load_model
from tqdm import trange
from time import sleep
from .train_types.helpers import create_attack_config, get_adversarial_attack
from PIL import Image
from blended_diffusion.optimization import DiffusionAttack
from utils_svces.Evaluator import Evaluator

def _prepare_targeted_translations(model_descriptions, imgs, target_list):
    num_datapoints = len(imgs)
    num_models = len(model_descriptions)

    max_num_targets = max([len(T) for T in target_list])
    perturbation_targets = torch.empty((num_datapoints, max_num_targets), dtype=torch.long).fill_(-1)
    for i in range(num_datapoints):
        datapoint_target_list = target_list[i]
        datapoint_target_vector = torch.empty(max_num_targets).fill_(-1)
        for j, val in enumerate(datapoint_target_list):
            datapoint_target_vector[j] = val
        perturbation_targets[i] = datapoint_target_vector

    all_models_perturbation_targets = torch.zeros((num_datapoints, num_models, max_num_targets), dtype=torch.long)
    for i in range(num_models):
        all_models_perturbation_targets[:, i, :] = perturbation_targets

    return all_models_perturbation_targets


def _compute_distances(a,b):
    diff = a - b
    diff_flat = diff.reshape(-1)
    l1 = torch.linalg.norm(diff_flat, ord=1.0)
    l2 = torch.linalg.norm(diff_flat, ord=2.0)
    return l1, l2

def _plot_diff_image(a,b, filepath):
    diff = (a - b).sum(2)
    min_diff_pixels = diff.min()
    max_diff_pixels = diff.max()
    min_diff_pixels = -max(abs(min_diff_pixels), max_diff_pixels)
    max_diff_pixels = -min_diff_pixels
    diff_scaled = (diff - min_diff_pixels) / (max_diff_pixels - min_diff_pixels)
    cm = plt.get_cmap('seismic')
    colored_image = cm(diff_scaled.numpy())
    pil_img = Image.fromarray(np.uint8(colored_image * 255.))
    pil_img.save(filepath)

def _plot_single_img(torch_img, filepath):
    pil_img = Image.fromarray(np.uint8(torch_img.numpy() * 255.))
    pil_img.save(filepath)

def _plot_counterfactuals(dir, model_name, model_checkpoint, original_imgs, original_probabilities, targets,
                          perturbed_imgs, perturbed_probabilities, radii, class_labels, filenames=None,
                          plot_single_images=False, show_distances=False):


    num_imgs = targets.shape[0]
    num_radii = len(radii)
    scale_factor = 1.8
    for img_idx in trange(num_imgs, desc=f'{model_name} {model_checkpoint} - Image write'):
        if filenames is None:
            single_img_dir = os.path.join(dir, f'{img_idx}')
        else:
            single_img_dir = os.path.join(dir, os.path.splitext(filenames[img_idx])[0])

        if plot_single_images:
            pathlib.Path(single_img_dir).mkdir(parents=True, exist_ok=True)

        img_targets = targets[img_idx,:]
        valid_target_idcs = torch.nonzero(img_targets != -1, as_tuple=False).squeeze(dim=1)
        num_targets =  len(valid_target_idcs)

        num_rows = num_targets
        num_cols = num_radii + 1
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
        if num_rows == 1:
            ax = np.expand_dims(ax, 0)
        img_original = original_imgs[img_idx, :].permute(1, 2, 0).cpu().detach()
        img_probabilities = original_probabilities[img_idx, :]
        img_confidence, img_prediction = torch.max(img_probabilities, dim=0)

        if plot_single_images:
            _plot_single_img(img_original, os.path.join(single_img_dir, 'original.png'))

        if num_targets == 2:
            title = f'{class_labels[img_targets[0]]}: {img_probabilities[img_targets[0]]:.2f}\n' \
                    f'{class_labels[img_targets[1]]}: {img_probabilities[img_targets[1]]:.2f}'
        else:
            title = f'{class_labels[img_prediction]}: {img_confidence:.2f}'

        # plot original:
        ax[0, 0].axis('off')
        ax[0, 0].set_title(title)
        ax[0, 0].imshow(img_original, interpolation='lanczos')

        for j in range(1, num_rows):
            ax[j, 0].axis('off')

        #plot counterfactuals
        for target_idx_idx in range(num_targets):
            target_idx = valid_target_idcs[target_idx_idx]
            for radius_idx in range(len(radii)):
                img = torch.clamp(perturbed_imgs[img_idx, target_idx, radius_idx].permute(1, 2, 0), min=0.0, max=1.0)
                img_target = targets[img_idx, target_idx]
                img_probabilities = perturbed_probabilities[img_idx, target_idx, radius_idx]

                l1, l2 = _compute_distances(img, img_original)
                target_conf = img_probabilities[img_target]

                ax[target_idx, radius_idx + 1].axis('off')
                ax[target_idx, radius_idx + 1].imshow(img, interpolation='lanczos')

                if num_targets == 2:
                    title = f'{class_labels[img_targets[0]]}: {img_probabilities[img_targets[0]]:.2f}\n' \
                            f'{class_labels[img_targets[1]]}: {img_probabilities[img_targets[1]]:.2f}'
                else:
                    title = f'{class_labels[img_target]}: {target_conf:.2f}\nl1: {l1:.3f}\nl2: {l2:.3f}'

                if show_distances:
                    pass

                ax[target_idx, radius_idx + 1].set_title(title)

                if plot_single_images:
                    _plot_single_img(img, os.path.join(single_img_dir, f'target_{target_idx}_radius_{radius_idx}.png'))
                    _plot_diff_image(img_original, img,
                                     os.path.join(single_img_dir, f'target_{target_idx}_radius_{radius_idx}_diff.png') )

        plt.tight_layout()
        if filenames is not None:
            fig.savefig(os.path.join(dir, f'{filenames[img_idx]}.png'))
            fig.savefig(os.path.join(dir, f'{filenames[img_idx]}.pdf'))
        else:
            fig.savefig(os.path.join(dir, f'{img_idx}.png'))
            fig.savefig(os.path.join(dir, f'{img_idx}.pdf'))

        plt.close(fig)

def _get_attack_config(eps, adv_attack_parameters):
    attack_config = create_attack_config(eps,
                                         adv_attack_parameters['steps'],
                                         adv_attack_parameters['stepsize'],
                                         adv_attack_parameters['norm'],
                                         pgd=adv_attack_parameters['attack_type'],
                                         normalize_gradient=True)
    return attack_config

def _inner_generation(original_imgs, perturbation_targets, model_descriptions, bs, class_labels, device, eval_dir,
                      dataset, adv_attack_parameters=None, diffusion_parameters=None, filenames=None,
                      plot_single_images=False, show_distanes=False, device_ids=None, verbose=False):

    assert not (adv_attack_parameters is None and diffusion_parameters is None)
    assert not (adv_attack_parameters is not None and diffusion_parameters is not None)
    if adv_attack_parameters is not None:
        radii = adv_attack_parameters['radii']
        use_diffusion = False
    else:
        radii = diffusion_parameters['lp_reg_weights']
        use_diffusion = True

    num_classes = len(class_labels)
    img_dimensions = original_imgs.shape[1:]
    num_targets = perturbation_targets.shape[2]
    num_radii = len(radii)
    num_imgs = len(original_imgs)

    with torch.no_grad():

        for model_idx in range(len(model_descriptions)):
            if np.isscalar(bs):
                model_bs = bs
            else:
                model_bs = bs[model_idx]

            type, model_folder, model_checkpoint, temperature, temp = model_descriptions[model_idx]
            print(f'{model_folder} {model_checkpoint} - bs {model_bs}')

            dir = f'{eval_dir}/{model_folder}_{model_checkpoint}/VisualCounterfactuals/'
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

            model = load_model(type, model_folder, model_checkpoint, temperature, device, load_temp=temp, dataset=dataset)
            if device_ids is not None and len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)
            model.eval()

            out_imgs = torch.zeros((num_imgs, num_targets, num_radii) + img_dimensions)
            out_probabilities = torch.zeros((num_imgs, num_targets, num_radii,num_classes))
            model_original_probabilities = torch.zeros((num_imgs, num_classes))

            n_batches = int(np.ceil(num_imgs / model_bs))

            if use_diffusion or config['pgd'] == 'afw':
                if use_diffusion:
                    att = DiffusionAttack(diffusion_parameters['hps'])
                else:
                    loss = 'ce-targeted-cfts-conf'
                    att = get_adversarial_attack(attack_config, model, loss, num_classes, args=diff_parameters['hps'], Evaluator=Evaluator)
                if self.args.second_classifier_type != -1:
                    print('setting model to second classifier')
                    model = att.second_classifier
                else:
                    model = att.classifier

            for batch_idx in trange(n_batches, desc=f'{model_folder} {model_checkpoint} - Batches progress'):
                sleep(0.1)
                batch_start_idx = batch_idx * model_bs
                batch_end_idx = min(num_imgs, (batch_idx + 1) * model_bs)

                batch_data = original_imgs[batch_start_idx:batch_end_idx, :]
                batch_targets = perturbation_targets[batch_start_idx:batch_end_idx, model_idx]

                orig_out = model(batch_data)
                with torch.no_grad():
                    orig_confidences = torch.softmax(orig_out, dim=1)
                    model_original_probabilities[batch_start_idx:batch_end_idx, :] = orig_confidences.detach().cpu()

                for radius_idx in range(len(radii)):
                    batch_data = batch_data.to(device)
                    batch_targets = batch_targets.to(device)

                    if use_diffusion:
                        att.args.lp_custom_value = diffusion_parameters['lp_reg_weights'][radius_idx]
                    elif config['pgd'] == 'afw':
                        att.eps = radii[radius_idx]
                    else:
                        eps = radii[radius_idx]
                        loss = 'ce-targeted-cfts-conf'
                        print('using loss', loss)
                        attack_config = _get_attack_config(eps, adv_attack_parameters)
                        att = get_adversarial_attack(attack_config, model, loss, num_classes, args=diff_parameters['hps'])

                    for target_idx in range(num_targets):
                        batch_targets_i = batch_targets[:, target_idx]
                        #use -1 as invalid index
                        valid_batch_targets = batch_targets_i != -1
                        num_valid_batch_targets = torch.sum(valid_batch_targets).item()
                        batch_adv_samples_i = torch.zeros_like(batch_data)
                        if num_valid_batch_targets > 0:
                            if use_diffusion:
                                batch_valid_adv_samples_i = att.perturb(batch_data[valid_batch_targets],
                                                                  batch_targets_i[valid_batch_targets], dir)[0].detach()
                            else:
                                if config['pgd'] == 'afw':
                                    att.classifier.T = torch.tensor([att.classifier.T], device=device).repeat(
                                        (batch_data[valid_batch_targets].shape[0], 1))
                                    batch_valid_adv_samples_i = att.perturb(batch_data[valid_batch_targets],
                                                                            batch_targets_i[valid_batch_targets],
                                                                            targeted=True).detach()

                                    print('setting temperature back to', temperature)
                                    att.classifier.T = temperature

                                else:
                                    batch_valid_adv_samples_i = att.perturb(batch_data[valid_batch_targets],
                                                                            batch_targets_i[valid_batch_targets],
                                                                            best_loss=True)[0].detach()

                            batch_adv_samples_i[valid_batch_targets] = batch_valid_adv_samples_i
                            with torch.no_grad():
                                batch_model_out_i = model(batch_adv_samples_i)
                                batch_probs_i = torch.softmax(batch_model_out_i, dim=1)

                            out_imgs[batch_start_idx:batch_end_idx, target_idx, radius_idx, :]\
                                = batch_adv_samples_i.cpu().detach()
                            out_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx, :]\
                                = batch_probs_i.cpu().detach()


            _plot_counterfactuals(dir, model_folder, model_checkpoint, original_imgs, model_original_probabilities,
                                  perturbation_targets[:, model_idx, :], out_imgs, out_probabilities, radii,
                                  class_labels, filenames=filenames, plot_single_images=plot_single_images,
                                  show_distances=show_distanes)

            out_dict = {'model_original_probabilities': model_original_probabilities,
                        'perturbation_targets': perturbation_targets[:, model_idx, :],
                        'out_probabilities': out_probabilities,
                        'radii': radii,
                        'class_labels': class_labels
                        }

            out_file = os.path.join(dir, 'info.pt')
            torch.save(out_dict, out_file)

#create counterfactuals for all models on all datapoints [N, 3, IMG_W, IMG_H] in the given classes
#target list should be a nested list where
def targeted_translations(model_descriptions, imgs, target_list, bs, class_labels, device, eval_dir, dataset,
                          adv_attack_parameters=None, diffusion_parameters=None,
                          show_distanes=False, filenames=None, device_ids=None):
    perturbation_targets =  _prepare_targeted_translations(model_descriptions, imgs, target_list)

    _inner_generation(imgs, perturbation_targets, model_descriptions, bs, class_labels, device, eval_dir, dataset,
                      adv_attack_parameters=adv_attack_parameters, diffusion_parameters=diffusion_parameters,
                      filenames=filenames, plot_single_images=True, show_distanes=show_distanes, device_ids=device_ids)
