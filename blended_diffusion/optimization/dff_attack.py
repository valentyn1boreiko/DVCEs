import io
import math
import os, inspect, sys
import pickle
from functools import partial

from pathlib import Path
import gc
import time
from blended_diffusion.optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR
from blended_diffusion.resizer import Resizer
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from blended_diffusion.utils_blended.metrics_accumulator import MetricsAccumulator
from blended_diffusion.utils_blended.model_normalization import ResizeWrapper, ResizeAndMeanWrapper
from blended_diffusion.utils_blended.video import save_video
import torch.nn as nn
import random
from blended_diffusion.optimization.augmentations import ImageAugmentations
from ResizeRight import resize_right
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss
from blended_diffusion.optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np

from blended_diffusion.guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
)

import matplotlib.pyplot as plt
from blended_diffusion.utils_blended.visualization import show_tensor_image, show_editied_masked_image

from utils_svces.get_config import get_config
from utils_svces.Evaluator import Evaluator


class EmptyWriter:
    def __init__(self):
        pass

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)

def cone_projection(grad_temp_1, grad_temp_2, deg, subspace_projection=False):

    if subspace_projection:
        grad_temp = []
        for i_image, grad_temp_1_image in enumerate(grad_temp_1):
            print(i_image)
            print((grad_temp_2[i_image].unsqueeze(0) * grad_temp_1_image[0]).shape, grad_temp_1_image[0].shape)
            grad_temp.append(sum([grad_temp_1_image_dimension * ((grad_temp_2[i_image].unsqueeze(0) * grad_temp_1_image_dimension).sum() / (grad_temp_1_image_dimension*grad_temp_1_image_dimension).sum()) for grad_temp_1_image_dimension in grad_temp_1_image]))
        del grad_temp_1

        print(grad_temp[0].shape)
        grad_temp = torch.stack(grad_temp, 0)
        print('subspace_proj shape', grad_temp.shape)
    else:
        angles_before = torch.acos(
            (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))

        grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
        grad_temp_1 = grad_temp_1 - ((grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1) ** 2)).view(
            grad_temp_1.shape[0], -1) * grad_temp_2
        grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
        radians = torch.tensor([deg], device=grad_temp_1.device).deg2rad()

        cone_projection = grad_temp_1 * torch.tan(radians) + grad_temp_2

        # second classifier is a non-robust one -
        # unless we are less than alpha degrees away - don't cone project
        grad_temp = grad_temp_2.clone()
        grad_temp[angles_before > radians] = cone_projection[angles_before > radians]

    return grad_temp

def _map_img(x):
    return 0.5 * (x + 1)

def _renormalize_gradient(grad, eps, small_const=1e-22):
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    grad_norm = torch.where(grad_norm < small_const, grad_norm + small_const, grad_norm)
    grad /= grad_norm
    grad *= eps.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    return grad, grad_norm


def compute_lp_dist(x, y, p):

    if int(p) == 1:
        lp_dist = torch.nn.functional.l1_loss(x, y)
    elif int(p) == 2:
        lp_dist = torch.nn.functional.mse_loss(x, y)
    else:
        lp_dist = torch.mean(abs(x - y)**p)
    return lp_dist

def compute_lp_gradient(diff, p, small_const=1e-12):
    if p < 1:
        grad_temp = (p * (diff.abs() + small_const) ** (

                    p - 1)) * diff.sign()
    else:
        grad_temp = (p * diff.abs() ** (p - 1)) * diff.sign()
    return grad_temp

def min_max_scale(tensor):
    tensor_ = tensor.clone()
    tensor_ -= tensor_.view(tensor.shape[0], -1).min(1)[0].view(-1, 1, 1, 1)
    tensor_ /= tensor_.view(tensor.shape[0], -1).max(1)[0].view(-1, 1, 1, 1)
    return tensor_

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  init_image_pil = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
  init_image_pil = init_image_pil.resize((224, 224), Image.LANCZOS)  # type: ignore
  image = TF.to_tensor(init_image_pil).unsqueeze(0)
  return image

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2 * (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

    if c2.nelement != 0:

        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)


        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)

def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def project_perturbation(perturbation, eps, p, center=None):
    if p in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        print('l2 renorm')
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    elif p in [1, 1.0, 'l1', 'L1', '1']:

        pert_normalized = L1_projection(center, perturbation, eps)
        return perturbation + pert_normalized

    else:
        raise NotImplementedError('Projection only supports l1, l2 and inf norm')

class DiffusionAttack():
    def __init__(self, args) -> None:
        self.args = args
        self.probs = None
        self.y = None
        self.writer = None
        self.small_const = 1e-12
        self.tensorboard_counter = 0
        self.verbose = args.verbose

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )


        self.device = self.args.device
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.num_classes = 1000
        self.model.load_state_dict(
            torch.load(
                f"checkpoints/256x256_diffusion_uncond.pt",
                map_location="cpu"
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)

        if args.device_ids is not None and len(args.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=args.device_ids)

        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()

        if self.args.clip_guidance_lambda != 0:
            if args.device_ids is not None and len(args.device_ids) > 1:
                self.model = nn.DataParallel(self.model, device_ids=args.device_ids)
                if self.args.clip_guidance_lambda != 0:
                    self.clip_model = (
                    nn.DataParallel(clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False), device_ids=args.device_ids)
                )
            else:
                self.clip_model = (
                    clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
                )
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        args.device = self.device
        self.classifier_config = get_config(args)
        self.imagenet_labels = self.classifier_config.data.class_labels

        evaluator = Evaluator(args, self.classifier_config, {}, None)


        self.classifier = evaluator.load_model(
            self.args.classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_1, interpolation=self.args.interpolation_int_1)
        )
        print('temp o resize o mean wrapper on')
        self.classifier.to(self.device)
        self.classifier.eval()

        if self.args.second_classifier_type != -1:
            self.second_classifier = evaluator.load_model(
                self.args.second_classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_2,
                                                              interpolation=self.args.interpolation_int_2)
            )

            self.second_classifier.to(self.device)
            self.second_classifier.eval()


        if self.args.third_classifier_type != -1:
            self.third_classifier = evaluator.load_model(
                self.args.third_classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_3,
                                                              interpolation=self.args.interpolation_int_3)
            )

            self.third_classifier.to(self.device)
            self.third_classifier.eval()

        ### ILVR resizers
        print("creating resizers...")

        down = lambda down_N : lambda tensor: resize_right.resize(tensor, 1 / down_N).to(self.device)
        up = lambda down_N : lambda tensor: resize_right.resize(tensor, down_N).to(self.device)
        self.resizers = (down, up)
        ### ILVR resizers


        self.clip_size = self.args.classifier_size_1

        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        if self.args.lpips_sim_lambda != 0:
            self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)


        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()
        self.metrics = {**{"L"+str(p_): lambda x1, x2, p_=p_: (x1-x2).view(x1.shape[0], -1).norm(p=p_, dim=1) for p_ in [1, 2]}}


    def _compute_layers(self, x, classifier, model_name='resnet50'):

        if model_name.lower() == 'resnet50':
            return [classifier._modules[module_name](x) for module_name in classifier._modules.keys() if module_name not in ['fc', 'global_pool']]

    def _gram_schmidt(self, vv):
        def projection(u, v):
            return (v * u).sum() / (u * u).sum() * u

        print(vv.shape)
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)
        uu[:, 0] = vv[:, 0].clone()
        for k in range(1, nk):
            vk = vv[:, k].clone()
            uk = 0
            for j in range(0, k):
                uj = uu[:, j].clone()
                uk = uk + projection(uj, vk)
            uu[:, k] = vk - uk
        for k in range(nk):
            uk = uu[:, k].clone()
            uu[:, k] = uk / uk.norm()
        return uu

    def _compute_probabilities(self, x, classifier,  permuted_logits_order=None):

        logits = classifier(_map_img(x))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if permuted_logits_order is not None:
            permuted_logits = torch.load('utils/'+permuted_logits_order)
            return log_probs[:, permuted_logits], probs[:, permuted_logits]
        else:
            return log_probs, probs

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def norms_per_image(self, x, y, tensor):
        return {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                enumerate(tensor)}

    def tensorboard_log_metrics(self, x, y, i):
        for name, metric in self.metrics.items():
            tensor = metric(x, self.init_image.add(1).div(2).clamp(0, 1))
            self.writer.add_scalars(f"Metrics/{name}", self.norms_per_image(x, y, tensor), i)

    def return_metrics_per_image(self, x):
        out = ['']*x.shape[0]

        for name, metric in self.metrics.items():
            tensor = metric(x, self.init_image.add(1).div(2).clamp(0, 1))
            out = [i_x[1]+f'{name}:{tensor[i_x[0]].item():.2f},' for i_x in enumerate(out)]
        return out

    def images_with_titles(self, imgs, titles):

        assert len(imgs) == len(titles)
        images = []
        for i, img in enumerate(imgs):
            fig = plt.figure(figsize=(14, 14))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.init_image[i].add(1).div(2).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy())
            images.append(plot_to_image(fig))

            fig = plt.figure(figsize=(14,14))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
            plt.title(titles[i], fontdict = {'fontsize' : 40})
            images.append(plot_to_image(fig))

        out = torch.cat(images, dim=0)
        print(out.shape)
        return out

    def clip_loss(self, x_in, text_embed):

        if self.mask is not None:
            masked_input = x_in
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2).clamp(0, 1)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        print('embeds shape', image_embeds.shape, text_embed.shape)
        dists = d_clip_loss(image_embeds, text_embed, use_cosine=True)

        print('clips dists are', dists.shape, dists)
        clip_loss = dists


        return clip_loss

    def unaugmented_clip_distance(self, x, text_embed):
        x = F.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.clip_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, text_embed, use_cosine=True)

        return dists.item()

    def edit_image_by_prompt(self, x, y, dir):


        if not self.verbose:
            self.writer = EmptyWriter()
        else:
            self.writer = CorrectedSummaryWriter(dir)



        self.text_batch = [self.imagenet_labels[y_el].split(',')[0] if y_el != -1 else 'flower' for y_el in y]

        if self.args.clip_guidance_lambda != 0:
            if y is not None:

                text_embeds = [self.clip_model.encode_text(
                    clip.tokenize(text_batch_).to(self.device)
                ).float() for text_batch_ in self.text_batch]
            else:
                self.text_batch = None
                text_embeds = self.clip_model.encode_text(
                   clip.tokenize(self.args.prompt).to(self.device)
                ).float()
            print('shape text_embed', len(text_embeds),  self.text_batch, len( self.text_batch))


        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        print('shapes x', x.shape[-1], self.model_config["image_size"])
        if x.shape[-1] != self.model_config["image_size"]:
            x = transforms.Resize(self.image_size)(x)
            print('shapes x after', x.shape)
        self.init_image = (x.to(self.device).mul(2).sub(1).clone().detach())

        loss_temp = torch.tensor(0.0).to(self.device)
        if self.args.second_classifier_type != -1:
            if self.args.projecting_cone:
                print('using cone projection, step0')
                logits = self.classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))
                logits2 = self.second_classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))
                if self.args.third_classifier_type != -1:
                    logits3 = self.third_classifier(
                        self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))

            else:
                print('using ensemble')
                logits = (0.5*self.classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1)).softmax(1) + 0.5*self.second_classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1)).softmax(1))
        else:
            logits = self.classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))


        if self.args.second_classifier_type != -1 and not self.args.projecting_cone:
            log_probs = logits.log()
        else:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        current_bs = len(x)
        loss_indiv = log_probs[
            range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
        for i in range(current_bs):
            # We want to average at the "augmentations level"
            loss_temp += loss_indiv[i:: current_bs].mean()
        print('shape loss', loss_indiv.shape)
        print('targets', y.shape, y)

        if self.args.second_classifier_type != -1:
            if self.args.projecting_cone:
                self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]
                self.init_probs_second_classifier = logits2[:current_bs].softmax(1)[
                    range(current_bs), y.view(-1)]
                if self.args.third_classifier_type != -1:
                    self.init_probs_third_classifier = logits3[:current_bs].softmax(1)[
                        range(current_bs), y.view(-1)]
            else:
                self.probs = logits[:current_bs][range(current_bs), y.view(-1)]
        else:
            self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]

        self.init_probs = self.probs
        self.tensorboard_counter += 1



        self.mask = None


        def cond_fn_blended(x, t, y=None, eps=None, variance=None):
            print('use blended', self.args.classifier_lambda, self.args.background_preservation_loss, self.args.l2_sim_lambda, self.args.lpips_sim_lambda)
            if self.args.prompt == "":
                return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)


                loss = torch.tensor(0)

                if self.args.clip_guidance_lambda != 0:
                    # ToDo: change text_embeds to incorporate only one image here
                    clip_loss = self.clip_loss(x_in, text_embeds) * self.args.clip_guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())


                if self.args.classifier_lambda != 0:

                    loss_temp = torch.tensor(0.0).to(self.device)


                    logits = self.classifier(self.image_augmentations(x_in).add(1).div(2).clamp(0, 1))


                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    loss_indiv = log_probs[
                        range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                    for i in range(current_bs):
                        # We want to average at the "augmentations level"
                        loss_temp += loss_indiv[i:: current_bs].mean()


                    self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]


                    self.y = y
                    classifier_loss = loss_temp * self.args.classifier_lambda

                    loss = loss - classifier_loss


                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.background_preservation_loss:
                    if self.mask is not None:
                        masked_background = x_in
                    else:
                        masked_background = x_in

                    if self.args.lpips_sim_lambda:
                        loss = (
                            loss
                            + self.lpips_model(masked_background, self.init_image).sum()
                            * self.args.lpips_sim_lambda
                        )
                    if self.args.l2_sim_lambda:
                        loss = (
                            loss
                            + mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                        )



                self.tensorboard_counter += 1

                return -torch.autograd.grad(loss, x)[0]



        def cond_fn_clean(x, t, y=None, eps=None):
            grad_out = torch.zeros_like(x)
            x = x.detach().requires_grad_()
            t = self.unscale_timestep(t)
            with torch.enable_grad():
                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                x_in = out["pred_xstart"]


            self.tensorboard_counter += 1
            # compute classifier gradient
            keep_denoising_graph = self.args.denoise_dist_input
            with torch.no_grad():




                if self.args.classifier_lambda != 0:
                    with torch.enable_grad():
                        print('before classifier')
                        log_probs_1, probs_1 = self._compute_probabilities(self.image_augmentations(x_in), self.classifier)

                        target_log_confs_1 = log_probs_1[range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]

                        grad_1 = torch.autograd.grad(target_log_confs_1.mean(), x, retain_graph=True)[0]
                        self.writer.add_images('gradients first classifier',
                                                min_max_scale(grad_1.abs().sum(1).unsqueeze(1)),
                                                self.tensorboard_counter
                                               )

                        if self.verbose:
                            print('maximizing prob_log', target_log_confs_1.shape, target_log_confs_1,
                                  probs_1)

                        if self.args.second_classifier_type != -1:

                            perturbed_images = self.image_augmentations(x_in)

                            if int(self.args.second_classifier_type) in [20]:
                                grad_2 = 0
                                for i in range(int(self.args.aug_num)):
                                    print(i, perturbed_images.shape, perturbed_images[i].shape)
                                    log_probs_2, probs_2 = self._compute_probabilities(perturbed_images[i].unsqueeze(0), self.second_classifier,
                                                                                   permuted_logits_order=None)
                                    target_log_confs_2 = log_probs_2[range(current_bs), y.view(-1)]
                                    grad_2 += target_log_confs_2.numel() * \
                                              torch.autograd.grad(target_log_confs_2.mean(), x,
                                                                  retain_graph=keep_denoising_graph)[0]
                                    if self.verbose:
                                        print('second classifier probs_log', probs_2[range(current_bs), y.view(-1)])
                                grad_2 /= (target_log_confs_2.numel() * self.args.aug_num)
                            else:
                                self.writer.add_images('augmented images',
                                                       _map_img(perturbed_images),
                                                       self.tensorboard_counter
                                                       )
                                log_probs_2, probs_2 = self._compute_probabilities(perturbed_images,
                                                                                   self.second_classifier,
                                                                                   permuted_logits_order=None)
                                if self.verbose:
                                    print('second classifier probs_log', probs_2[range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)])
                                target_log_confs_2 = log_probs_2[range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                                grad_2 = \
                                    torch.autograd.grad(target_log_confs_2.mean(), x,
                                                        retain_graph=keep_denoising_graph)[0]

                            self.writer.add_images('gradients second classifier',
                                                   min_max_scale(grad_2.abs().sum(1).unsqueeze(1)),
                                                   self.tensorboard_counter
                                                   )
                            time_start = time.time()


                            grad_class = cone_projection(grad_1.view(x.shape[0], -1).cpu(),
                                                         grad_2.view(x.shape[0], -1).cpu(),
                                                         self.args.deg_cone_projection,
                                                         subspace_projection=False).view_as(grad_2).to(self.device)
                            print('projection_time', time.time() - time_start)
                            print('cone projection dist after', (grad_class - grad_2).norm(p=2))
                            if self.args.third_classifier_type != -1:


                                grad_class_third = cone_projection(grad_1.view(x.shape[0], -1),
                                                             (grad_3).view(x.shape[0], -1),
                                                             self.args.deg_cone_projection).view_as(grad_1)

                                grad_class -= grad_class_third
                        else:
                            grad_class = torch.autograd.grad(target_log_confs_1.mean(), x,
                                                             retain_graph=keep_denoising_graph)[0]

                    if self.args.enforce_same_norms:
                        grad_, norm_ = _renormalize_gradient(grad_class, eps)

                        grad_class = self.args.classifier_lambda * grad_

                    else:
                        grad_class *= self.args.classifier_lambda

                    grad_out += grad_class

                # distance gradients
                if self.args.lp_custom:
                    if not keep_denoising_graph:
                        print('not denoising_reguarization')
                        diff = x_in - self.init_image
                        lp_grad = compute_lp_gradient(diff, self.args.lp_custom)
                    else:
                        print('denoising_reguarization, new lpdist')
                        with torch.enable_grad():
                            lp_dist = compute_lp_dist(x_in, self.init_image, self.args.lp_custom)
                            lp_grad = torch.autograd.grad(lp_dist, x)[0]

                    if self.args.quantile_cut != 0:
                        pass

                    if self.args.enforce_same_norms:
                        print('enforcing same norms...')
                        grad_, norm_ = _renormalize_gradient(lp_grad, eps)

                        lp_grad = self.args.lp_custom_value * grad_



                    else:
                        lp_grad *= self.args.lp_custom_value

                    grad_out -= lp_grad

                if self.args.layer_reg:
                    if not keep_denoising_graph:
                        diff = 0
                        for x_in_layer, init_image_layer in zip(self._compute_layers(x_in, self.classifier), self._compute_layers(self.init_image, self.classifier)):
                            diff += compute_lp_gradient(x_in_layer-init_image_layer, self.args.layer_reg)
                    else:
                        with torch.enable_grad():
                            diff = self._compute_layers(x_in, self.classifier) - self._compute_layers(self.init_image, self.classifier)
                            lp_dist = compute_lp_dist(diff, self.args.layer_reg)
                            lp_grad = torch.autograd.grad(lp_dist.mean(), x)[0]
                    if self.args.enforce_same_norms:
                        print('enforcing same norms...')

                        grad_, norm_ = _renormalize_gradient(lp_grad, eps)

                        lp_grad = self.args.layer_reg_value * grad_



                    else:
                        lp_grad *= self.args.layer_reg_value

                    grad_out -= lp_grad

            return grad_out


        @torch.no_grad()
        def postprocess_fn(out, t):

            if self.mask is not None:

                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])

                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)


            return out

        targets_classifier = y

        if self.args.gen_type == 'ddim':
            gen_func = self.diffusion.ddim_sample_loop_progressive
        elif self.args.gen_type == 'p_sample':
            gen_func = self.diffusion.p_sample_loop_progressive
        else:
            raise ValueError(f'Generation type {self.args.gen_type} is not implemented.')

        samples = gen_func(
            self.model,
            (
                current_bs * len(self.args.device_ids),
                3,
                self.model_config["image_size"],
                self.model_config["image_size"],
            ),
            clip_denoised=False,
            model_kwargs={

                "y": torch.tensor(targets_classifier, device=self.device, dtype=torch.long)
            },
            cond_fn=cond_fn_blended if self.args.use_blended else cond_fn_clean,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=self.init_image if not self.args.not_use_init_image else None,
            postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
            randomize_class=False,
            resizers=self.resizers,
            range_t=self.args.range_t,
            eps_project=self.args.eps_project,
            ilvr_multi=self.args.ilvr_multi,
            seed=self.args.seed

        )

        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
        print('num total steps is', total_steps)
        max_probs = self.init_probs * 0
        sample_final = self.init_image
        print('before loop')
        for i, sample in enumerate(samples):
            print(i, max_probs)

            if i == total_steps:
                sample_final = sample["pred_xstart"]


        self.writer.flush()
        self.writer.close()
        return _map_img(sample_final).clamp(0, 1)

    def perturb(self, x, y, dir):

        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        self.tensorboard_counter = 0
        adv_best = self.edit_image_by_prompt(x, y, dir)
        return [adv_best]
