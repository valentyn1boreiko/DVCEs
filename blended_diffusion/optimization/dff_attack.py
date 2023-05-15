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
from torch.nn.functional import mse_loss
from blended_diffusion.optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np

#from blended_diffusion.CLIP import clip
from blended_diffusion.guided_diffusion.guided_diffusion.script_util import (
    #NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    #add_dict_to_argparser,
    #args_to_dict
)



import matplotlib.pyplot as plt
from blended_diffusion.utils_blended.visualization import show_tensor_image, show_editied_masked_image

#ACSM_dir = '/mnt/SHARED/valentyn/ACSM'
##ACSM_dir = '/scratch/vboreiko87/projects/ACSM'

##sys.path.insert(0, ACSM_dir)

##print(sys.path)

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

def cone_projection(grad_temp_1, grad_temp_2, deg):
    angles_before = torch.acos(
        (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))
    ##print('angle before', angles_before)

    grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
    grad_temp_1 = grad_temp_1 - ((grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1) ** 2)).view(
        grad_temp_1.shape[0], -1) * grad_temp_2
    grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
    # cone_projection = grad_temp_1 + grad_temp_2 45 deg
    radians = torch.tensor([deg], device=grad_temp_1.device).deg2rad()
    ##print('angle after', radians, torch.acos((grad_temp_1*grad_temp_2).sum(1) / (grad_temp_1.norm(p=2,dim=1) * grad_temp_2.norm(p=2,dim=1))))

    cone_projection = grad_temp_1 * torch.tan(radians) + grad_temp_2

    # second classifier is a non-robust one -
    # unless we are less than 45 degrees away - don't cone project
    grad_temp = grad_temp_2.clone()
    loop_projecting = time.time()
    grad_temp[angles_before > radians] = cone_projection[angles_before > radians]

    return grad_temp

def _map_img(x):
    return 0.5 * (x + 1)


def _renormalize_gradient(grad, eps, small_const=1e-22):
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    #print('grad norm is', grad_norm)
    grad_norm = torch.where(grad_norm < small_const, grad_norm+small_const, grad_norm)
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
  #image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  #image = tf.expand_dims(image, 0)
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
    # u = torch.min(u, epsinf - torch.clone(y).abs())
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

        # print(c2.shape, lb.shape)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            # print(ind3.shape)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            # print(lb, ub)
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
        ##pert_normalized = project_onto_l1_ball(perturbation, eps)
        ##return pert_normalized
        pert_normalized = L1_projection(center, perturbation, eps)
        return perturbation + pert_normalized
    #elif p in ['LPIPS']:
    #    pert_normalized = project_onto_LPIPS_ball(perturbation, eps)
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

        # Load models
        ##self.device = torch.device(
        ##    f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        ##)

        ##
        self.device = self.args.device
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.num_classes = 1000
        self.model.load_state_dict(
            torch.load(
                "checkpoints/256x256_diffusion_uncond.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
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
        #pickle.dump([args, self.classifier_config], open("Madry_ep3_args.pickle", "wb"))

        """
        print("loading classifier...")
        args.image_size = self.args.model_output_size
        args.classifier_use_fp16 = self.model_config["use_fp16"]
        self.classifier = create_classifier(**classifier_defaults()) #**args_to_dict(args, classifier_defaults().keys()))

        self.classifier.load_state_dict(
            torch.load("checkpoints/256x256_classifier.pt"
                if self.args.model_output_size == 256
                else None,
                map_location="cpu")
        )
        self.classifier.to(self.device)
        #if args.classifier_use_fp16:
        #    self.classifier.convert_to_fp16()
        self.classifier.eval()

        """
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

            #self.second_classifier = classifier
            self.second_classifier.to(self.device)
            self.second_classifier.eval()


        if self.args.third_classifier_type != -1:
            self.third_classifier = evaluator.load_model(
                self.args.third_classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_3,
                                                              interpolation=self.args.interpolation_int_3)
            )

            #self.second_classifier = classifier
            self.third_classifier.to(self.device)
            self.third_classifier.eval()

        ### ILVR resizers
        print("creating resizers...")
        #assert math.log(self.args.down_N, 2).is_integer()

        #shape = (self.args.batch_size, 3, self.model_config["image_size"], self.model_config["image_size"])
        #shape_d = (self.args.batch_size, 3, int(self.model_config["image_size"] / self.args.down_N),
        #           int(self.model_config["image_size"] / self.args.down_N))
        down = lambda down_N : lambda tensor: resize_right.resize(tensor, 1 / down_N).to(self.device) #lambda down_N : Resizer(shape, 1 / down_N).to(self.device) #Resizer(shape, 1 / self.args.down_N).to(self.device)
        up = lambda down_N : lambda tensor: resize_right.resize(tensor, down_N).to(self.device) #lambda down_N: Resizer(shape_d, down_N).to(self.device) #Resizer(shape_d, self.args.down_N).to(self.device)
        self.resizers = (down, up)
        ### ILVR resizers

        #try:
        #    self.clip_size = self.clip_model.visual.input_resolution
        #except Exception as err:
        #    print(str(err))
        #    self.clip_size = self.clip_model.module.visual.input_resolution

        self.clip_size = self.args.classifier_size_1

        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        if self.args.lpips_sim_lambda != 0:
            self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()
        self.metrics = {**{"L"+str(p_): lambda x1, x2, p_=p_: (x1-x2).view(x1.shape[0], -1).norm(p=p_, dim=1) for p_ in [1, 2]}}
        #               **{"LPIPS": lambda x1, x2: self.lpips_model(x1, x2),
        #                "MSSIM": lambda x1, x2: ms_ssim(x1, x2, data_range=1, size_average=False),
        #                "SSIM": lambda x1, x2: ssim(x1, x2, data_range=1, size_average=False)}}

    def _compute_probabilities(self, x, classifier):
        logits = classifier(_map_img(x))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, probs

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def norms_per_image(self, x, y, tensor):
        return {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                enumerate(tensor)}
        #                                   enumerate(tensor.view(x.shape[0], -1).norm(p=2, dim=1))}

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
        clip_loss = torch.tensor(0)

        if self.mask is not None:
            masked_input = x_in #* self.mask
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2).clamp(0, 1)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        print('embeds shape', image_embeds.shape, text_embed.shape)
        dists = d_clip_loss(image_embeds, text_embed, use_cosine=True)

        print('clips dists are', dists.shape, dists)
        clip_loss = dists

        # We want to sum over the averages
        #for i in range(self.args.batch_size):
        #    # We want to average at the "augmentations level"
        #    clip_loss = clip_loss + dists[i:: self.args.batch_size].mean()

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
        self.init_image = (x.to(self.device).mul(2).sub(1).clone())


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
            #logits = self.classifier(self.image_augmentations(self.init_image, variance=1),
            #                        torch.tensor([0.0]).to(self.device))
            logits = self.classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))


        if self.args.second_classifier_type != -1 and not self.args.projecting_cone:
            log_probs = logits.log()
        else:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # loss_indiv = log_probs[range(len(logits)), y.view(-1)]
        current_bs = len(x)
        loss_indiv = log_probs[
            range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
        for i in range(current_bs):
            # We want to average at the "augmentations level"
            loss_temp += loss_indiv[i:: current_bs].mean()
        print('shape loss', loss_indiv.shape)
        print('targets', y.shape, y)
        print('probs', logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)])

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
        probs_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                           enumerate(self.probs)}

        #self.writer.add_scalars('Probs,classifier', probs_per_image, self.tensorboard_counter)
        self.init_probs = self.probs
        self.tensorboard_counter+=1


        self.mask = None


        #def cond_fn(x, t, only_cond_fn, variance, y=None):
        def cond_fn_blended(x, t, y=None, eps=None, variance=None):
            print('use blended')
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
                # x_in = out["pred_xstart"]

                loss = torch.tensor(0)

                if self.args.clip_guidance_lambda != 0:
                    # ToDo: change text_embeds to incorporate only one image here
                    clip_loss = self.clip_loss(x_in, text_embeds) * self.args.clip_guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())


                if self.args.classifier_lambda != 0:

                    loss_temp = torch.tensor(0.0).to(self.device)
                    #print('x2', x.norm(p=2))#, masked_background)


                    logits = self.classifier(self.image_augmentations(x_in).add(1).div(2).clamp(0, 1))


                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    loss_indiv = log_probs[
                        range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                    for i in range(current_bs):
                        # We want to average at the "augmentations level"
                        loss_temp += loss_indiv[i:: current_bs].mean()

                        # += (loss_indiv[i:: current_bs].mean() + log_probs_extra[i].mean()) / 10
                    #print('loss is', loss_temp)
                    #print(
                    #'probs', logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)])
                    #print('dist', (self.init_image - x).view(current_bs, -1).norm(p=2, dim=1))
                    self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]


                    self.y = y
                    classifier_loss = loss_temp * self.args.classifier_lambda

                    loss = loss - classifier_loss
                    #self.metrics_accumulator.update_metric("classifier_loss", classifier_loss.item())


                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.background_preservation_loss:
                    if self.mask is not None:
                        masked_background = x_in #* (1 - self.mask)
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

                #titles = self.return_metrics_per_image(
                #    x.add(1).div(2).clamp(0, 1))
                #assert len(self.init_probs) == len(self.probs) == len(titles)

                #titles = [f'p_i:{self.init_probs[i_x[0]]:.2f},p_e:{self.probs[i_x[0]]:.2f},' + i_x[1].replace('L2',
                #                                                                                                  '\nL2') + f'target:{self.text_batch[i_x[0]]}'
                #              for i_x in enumerate(titles)]
                #self.writer.add_images('images for classifier', self.images_with_titles(
                #    x.add(1).div(2).clamp(0, 1), titles),
                #                       self.tensorboard_counter)

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

            #self.writer.add_images('images denoised',
            #                       x.add(1).div(2).clamp(0, 1)[:current_bs],
            #                       self.tensorboard_counter)

            #self.writer.add_images('images denoised',
            #                   x_in.add(1).div(2).clamp(0, 1)[:current_bs],
            #                       self.tensorboard_counter)

            #self.tensorboard_counter += 1
            # compute classifier gradient
            keep_denoising_graph = self.args.denoise_dist_input
            with torch.no_grad():
                if self.args.classifier_lambda != 0:
                    with torch.enable_grad():
                        log_probs_1, probs_1 = self._compute_probabilities(x_in, self.classifier)
                        target_log_confs_1 = log_probs_1[range(current_bs), y.view(-1)]

                        if self.args.second_classifier_type != -1:
                            log_probs_2, probs_2 = self._compute_probabilities(x_in, self.second_classifier)
                            target_log_confs_2 = log_probs_2[range(current_bs), y.view(-1)]

                            grad_1 = torch.autograd.grad(target_log_confs_1.mean(), x, retain_graph=True)[0]
                            grad_2 = \
                            torch.autograd.grad(target_log_confs_2.mean(), x, retain_graph=keep_denoising_graph)[0]

                            grad_class = cone_projection(grad_1.view(x.shape[0], -1),
                                                         grad_2.view(x.shape[0], -1),
                                                         self.args.deg_cone_projection).view_as(grad_1)
                        else:
                            grad_class = torch.autograd.grad(target_log_confs_1.mean(), x,
                                                             retain_graph=keep_denoising_graph)[0]

                    if self.args.enforce_same_norms:
                        grad_, norm_ = _renormalize_gradient(grad_class, eps)
                        #norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                        #                   enumerate(grad_class.view(x.shape[0], -1).norm(p=2, dim=1))}

                        #print('norms before classifier', norms_per_image)

                        #self.writer.add_scalars(f'Gradient norms before normalization/Classifier norm', norms_per_image,
                        #                        self.tensorboard_counter)

                        grad_class = self.args.classifier_lambda * grad_

                        #norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                        #                   enumerate(grad_class.view(x.shape[0], -1).norm(p=2, dim=1))}

                        #self.writer.add_scalars(f'Gradient norms after normalization/Classifier norm', norms_per_image,
                        #                        self.tensorboard_counter)
                    else:
                        grad_class *= self.args.classifier_lambda

                    grad_out += grad_class

                # distance gradients
                if self.args.lp_custom: # and self.args.range_t < self.tensorboard_counter:
                    if not keep_denoising_graph:
                        diff = x_in - self.init_image
                        lp_grad = compute_lp_gradient(diff, self.args.lp_custom)
                    else:
                        with torch.enable_grad():
                            diff = x_in - self.init_image
                            lp_dist = compute_lp_dist(diff, self.args.lp_custom)
                            lp_grad = torch.autograd.grad(lp_dist.mean(), x)[0]
                    if self.args.quantile_cut != 0:
                        pass

                    if self.args.enforce_same_norms:
                        #norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                        #                   enumerate(diff.view(norm_.shape[0], -1).norm(p=1, dim=1))}
                        #self.writer.add_scalars(f'Distances/L1 norm', norms_per_image,
                        #                        self.tensorboard_counter)

                        grad_, norm_ = _renormalize_gradient(lp_grad, eps)
                        #norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                        #                   enumerate(norm_.view(norm_.shape[0], -1))}
                        #print('norms before lp', norms_per_image)
                        #self.writer.add_scalars(f'Gradient norms before normalization/L1 norm', norms_per_image,
                        #                        self.tensorboard_counter)

                        lp_grad = self.args.lp_custom_value * grad_

                        #norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                        #                   enumerate(lp_grad.view(x.shape[0], -1).norm(p=2, dim=1))}

                        #self.writer.add_scalars(f'Gradient norms after normalization/L1 norm', norms_per_image,
                        #                        self.tensorboard_counter)

                    else:
                        lp_grad *= self.args.lp_custom_value

                    grad_out -= lp_grad

                if self.args.mssim_lambda > 0:
                    pass

                if self.args.ssim_lambda > 0:
                    pass

            #self.tensorboard_counter += 1

            return grad_out

        def cond_fn(x, t, y=None, eps=None, variance=None):

            #start_cond_fn = time.time()
            if self.args.prompt == "":
                return torch.zeros_like(x)

            grad_out = torch.zeros_like(x)

            #skip_backgrond = (int(self.args.timestep_respacing) - self.args.skip_timesteps - self.tensorboard_counter < self.args.range_t)
            #print('skip background loss', (int(self.args.timestep_respacing) - self.args.skip_timesteps - self.tensorboard_counter < self.args.range_t), self.tensorboard_counter)

            with torch.enable_grad():
                x = x.detach().requires_grad_()


                #print('t1', t)
                t = self.unscale_timestep(t)
                #print('t1', t)

                #self.writer.add_images('images before denoising 1',
                #                       x.add(1).div(2).clamp(0, 1)[:current_bs],
                #                       self.tensorboard_counter)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                x_in = out["pred_xstart"]
                #if self.mask is not None:
                #    masked_background = x_in  # * (1 - self.mask)
                #else:


            #pred_xstart_cache = out["pred_xstart"].deepcopy()
            # print('out1', x.norm(p=2), out["pred_xstart"].norm(p=2), out["pred_xstart"])
            #fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
            #fac_cached = fac.copy()  # .clone()
            #x.grad.data.zero_()

            #out = self.diffusion.p_mean_variance(
            #    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}, use_cached_output=True
            #)
            #out["pred_xstart"] = pred_xstart_cache.clone()
            #print('out2', x.norm(p=2), out["pred_xstart"].norm(p=2), out["pred_xstart"])

            #fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
            #fac = fac_cached.copy()#.clone()
            #x_in = out["pred_xstart"] * fac + x * (1 - fac)
            #print("x_in_2", x_in.norm(p=2), x_in, fac)
            #x_in = out["pred_xstart"]
            #if self.mask is not None:
            #    masked_background = x_in  # * (1 - self.mask)
            #else:
            #    masked_background = x_in
            #loss = torch.tensor(0)
            if self.args.clip_guidance_lambda != 0:

                clip_losses = [self.clip_loss(masked_background_, text_embeds[i]) for i, masked_background_ in enumerate(x_in)] #* self.args.clip_guidance_lambda

                #print('clip losses are', clip_losses)
                #self.probs = [1-x for x in clip_losses]

                clip_loss = sum(clip_losses)
                grad_temp = torch.autograd.grad(clip_loss, x)[0]#.detach()

                grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)

                if self.args.enforce_same_norms:
                    # alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                    grad_temp *= self.args.clip_guidance_lambda * eps.view(x.shape[0], -1).norm(p=2, dim=1).view(
                        x.shape[0], 1, 1, 1)

                    #norms_per_image = {str(i_val[0]): i_val[1].item() for i_val in
                    #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}

                    # self.writer.add_scalars(f'Gradient norms after normalization/Classifier norm', norms_per_image,
                    #                        self.tensorboard_counter)
                    #print('enforce_same_norms: norms CLIP', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1),
                    #      eps.view(x.shape[0], -1).norm(p=2, dim=1))
                else:
                    grad_temp *= self.args.clip_guidance_lambda
                #print('gradient CLIP norms after', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))

                #self.writer.add_images('gradients of CLIP',
                #                       grad_temp.abs().sum(1).unsqueeze(1),
                #                       self.tensorboard_counter)

                grad_out -= grad_temp
                #loss = loss + clip_loss
                #self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
            if self.args.classifier_lambda != 0:
                #classifier_update = time.time()
                with torch.enable_grad():
                    loss_temp = torch.tensor(0.0).to(self.device)
                    ##print('x2', x.norm(p=2))#, masked_background)
                    #computing_logits_log_probs = time.time()
                    if self.args.second_classifier_type != -1:
                        #print('using ensemble')
                        if self.args.projecting_cone:
                            logits1 = self.classifier(self.image_augmentations(x_in).add(1).div(2).clamp(0, 1))
                            logits2 = self.second_classifier(self.image_augmentations(x_in).add(1).div(2).clamp(0, 1))
                            if self.args.third_classifier_type:
                                logits3 = self.third_classifier(
                                    self.image_augmentations(x_in).add(1).div(2).clamp(0, 1))
                        else:
                            logits = (0.5*self.classifier(
                            self.image_augmentations(x_in).add(1).div(2).clamp(0, 1)).softmax(
                            1) + 0.5*self.second_classifier(
                            self.image_augmentations(x_in).add(1).div(2).clamp(0, 1)).softmax(1))

                    else:
                        logits = self.classifier(self.image_augmentations(x_in).add(1).div(2).clamp(0, 1))
                        #logits = self.classifier(x, t)
                        #logits = self.classifier(self.image_augmentations(x, variance=0.01), torch.tensor([t[0].item()], device=self.device))



                    #logits = self.classifier(self.image_augmentations(masked_background).add(1).div(2))
                    #logits = self.classifier(masked_background.add(1).div(2))
                    #self.writer.add_images('images before classifier', masked_background.add(1).div(2).clamp(0, 1)[:self.args.batch_size], self.tensorboard_counter)
                    #self.writer.add_images('images before classifier',
                    #                       x.add(1).div(2).clamp(0, 1)[:self.args.batch_size],
                    #                       self.tensorboard_counter)
                    if self.args.second_classifier_type != -1:
                        if self.args.projecting_cone:
                            log_probs_1 = torch.nn.functional.log_softmax(logits1, dim=-1)
                            log_probs_2 = torch.nn.functional.log_softmax(logits2, dim=-1)
                            if self.args.third_classifier_type:
                                log_probs_3 = torch.nn.functional.log_softmax(logits3, dim=-1)
                        else:
                            log_probs = logits.log()

                    else:
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                        #log_probs_extra = torch.nn.functional.log_softmax(self.classifier(x.clone() + variance * torch.randn_like(x), t), dim=-1)
                        #for _ in range(8):
                        #    log_probs_extra += torch.nn.functional.log_softmax(self.classifier(x.clone() + variance * torch.randn_like(x), t), dim=-1)
                    #print('computing logits and softmax time', time.time() - computing_logits_log_probs)
                    # loss_indiv = log_probs[range(len(logits)), y.view(-1)]
                    if self.args.projecting_cone and self.args.second_classifier_type != -1:

                        #projecting_cone = time.time()

                        loss_temp_1 = torch.tensor(0.0).to(self.device)
                        loss_temp_2 = torch.tensor(0.0).to(self.device)
                        if self.args.third_classifier_type:
                            loss_temp_3 = torch.tensor(0.0).to(self.device)

                        loss_indiv_1 = log_probs_1[
                            range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                        loss_indiv_2 = log_probs_2[
                            range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                        if self.args.third_classifier_type:
                            loss_indiv_3 = log_probs_3[
                                range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                        for i in range(current_bs):
                            # We want to average at the "augmentations level"
                            loss_temp_1 += loss_indiv_1[i:: current_bs].mean()
                            loss_temp_2 += loss_indiv_2[i:: current_bs].mean()
                            if self.args.third_classifier_type:
                                loss_temp_3 += loss_indiv_3[i:: current_bs].mean()

                            # += (loss_indiv[i:: current_bs].mean() + log_probs_extra[i].mean()) / 10
                        ##print('loss is', loss_temp)
                        ##print(
                        ##    'probs', logits1[:current_bs].softmax(1)[range(current_bs), y.view(-1)])
                        ##print('dist', (self.init_image - x).view(current_bs, -1).norm(p=2, dim=1))


                        #self.probs = logits1[:current_bs].softmax(1)[
                        #        range(current_bs), y.view(-1)]

                        #self.probs_second_classifier = logits2[:current_bs].softmax(1)[
                        #    range(current_bs), y.view(-1)]
                        #if self.args.third_classifier_type:
                        #    self.probs_third_classifier = logits3[:current_bs].softmax(1)[
                        #        range(current_bs), y.view(-1)]

                        # self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]
                        # probs_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                        #                   enumerate(self.probs)}

                        # self.writer.add_scalars('Probs,classifier', probs_per_image, self.tensorboard_counter)

                        #self.y = y


                        #grad1_time = time.time()
                        grad_temp_1 = torch.autograd.grad(loss_temp_1, x, retain_graph=True)[0].view(x.shape[0], -1)  # .detach()
                        #print('grad1 time', time.time() - grad1_time)
                        if self.args.third_classifier_type:
                            #grad3_time = time.time()
                            grad_temp_3 = torch.autograd.grad(loss_temp_3, x, retain_graph=True)[0].view(x.shape[0],
                                                                                                         -1)  # .detach()

                            #print('grad3 time', time.time() - grad3_time)
                        #grad2_time = time.time()
                        grad_temp_2 = torch.autograd.grad(loss_temp_2, x)[0].view(x.shape[0], -1)
                        if self.args.third_classifier_type:
                            grad_temp_2 -= grad_temp_3
                            #pass


                        #print('grad2 time', time.time() - grad2_time)
                        with torch.no_grad():
                            angles_before = torch.acos((grad_temp_1*grad_temp_2).sum(1) / (grad_temp_1.norm(p=2,dim=1) * grad_temp_2.norm(p=2,dim=1)))
                            ##print('angle before', angles_before)

                            grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(x.shape[0], -1)
                            grad_temp_1 = grad_temp_1 - ((grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1)**2)).view(x.shape[0], -1) * grad_temp_2
                            grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(x.shape[0], -1)
                            #cone_projection = grad_temp_1 + grad_temp_2 45 deg
                            radians = torch.tensor([self.args.deg_cone_projection], device=x.device).deg2rad()
                            ##print('angle after', radians, torch.acos((grad_temp_1*grad_temp_2).sum(1) / (grad_temp_1.norm(p=2,dim=1) * grad_temp_2.norm(p=2,dim=1))))

                            cone_projection = grad_temp_1*torch.tan(radians) + grad_temp_2

                            #second classifier is a non-robust one -
                            # unless we are less than 45 degrees away - don't cone project
                            grad_temp = grad_temp_2.clone()
                            loop_projecting = time.time()
                            grad_temp[angles_before > radians] = cone_projection[angles_before > radians]
                            #print('vectorized projecting time', time.time() - loop_projecting)
                            ##print('norms after', grad_temp_2.norm(p=2,dim=1), grad_temp.norm(p=2,dim=1), cone_projection.norm(p=2, dim=1))



                            ##angles_after = torch.acos((grad_temp * grad_temp_2).sum(1) / (
                            ##            (grad_temp).norm(p=2, dim=1) * grad_temp_2.norm(p=2,dim=1)))
                            ##print('angle after projection', angles_after)

                            grad_temp = grad_temp.view(x.shape)
                            #print('projecting cone time', time.time() - projecting_cone)

                    else:
                        loss_indiv = log_probs[
                            range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                        for i in range(current_bs):
                            # We want to average at the "augmentations level"
                            loss_temp += loss_indiv[i:: current_bs].mean()

                            # += (loss_indiv[i:: current_bs].mean() + log_probs_extra[i].mean()) / 10
                        #print('loss is', loss_temp)
                        #print(
                        #'probs', logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)])
                        #print('dist', (self.init_image - x).view(current_bs, -1).norm(p=2, dim=1))

                        #if self.args.second_classifier_type != -1:
                        #    self.probs = logits[:current_bs][range(current_bs), y.view(-1)]
                        #else:
                        #    self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]

                        #self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]
                        #probs_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                        #                   enumerate(self.probs)}

                        #self.writer.add_scalars('Probs,classifier', probs_per_image, self.tensorboard_counter)


                        #self.y = y
                        classifier_loss = loss_temp #* self.args.classifier_lambda

                        grad_temp = torch.autograd.grad(classifier_loss, x)[0]#.detach()
                #del classifier_loss, loss_temp, loss_indiv, log_probs, logits
                #classifier_loss.backward(retain_graph=True, inputs=x)
                #grad_temp = x.grad.data.clone()
                #norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                #print('gradient classifier norms before', norms_per_image)
                #self.writer.add_scalars(
                #    'Gradients of norm' + ', seed:' + str(
                #        self.args.seed) + ', l1.5 Regularization, class:/classifier', norms_per_image,
                #    self.tensorboard_counter)
                #normalization_same_norms = time.time()
                grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)
                ##print('classifier normm', grad_temp.norm(p=2), self.args.classifier_lambda)
                if self.args.enforce_same_norms:
                    # alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                    grad_temp *= self.args.classifier_lambda * eps.view(x.shape[0], -1).norm(p=2, dim=1).view(
                        x.shape[0], 1, 1, 1)

                    ##norms_per_image = {str(i_val[0]): i_val[1].item() for i_val in
                    ##                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}

                    #self.writer.add_scalars(f'Gradient norms after normalization/Classifier norm', norms_per_image,
                    #                        self.tensorboard_counter)
                    #print('enforce_same_norms: norms classifier', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1),
                    #      eps.view(x.shape[0], -1).norm(p=2, dim=1))
                else:
                    grad_temp *= self.args.classifier_lambda
                #print('normalization and same norms time', time.time() - normalization_same_norms)
                #print('gradient classifier norms after', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))

                ##print('classifier normm', grad_temp.norm(p=2), self.args.classifier_lambda)
                ##self.writer.add_images('gradients of classifier',
                ##    grad_temp.abs().sum(1).unsqueeze(1),
                ##                       self.tensorboard_counter)
                grad_out += grad_temp
                #print('classifier update time', time.time() - classifier_update)
                #del grad_temp
                #grad_temp = torch.autograd.grad(classifier_loss, x)[0].detach()

                #norms_per_image = {str(i_val[0]): i_val[1].item() for i_val in
                #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                #print('gradient classifier norms before', norms_per_image)
                #self.writer.add_scalars(
                #    'Gradients of norm' + ', seed:' + str(self.args.seed) + ', l1.5 Regularization, class:' +
                #    class_labels[y[0]] + '/classifier', norms_per_image, self.tensorboard_counter)

                # grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)
                #grad_temp *= self.args.classifier_lambda
                #print('gradient classifier norms after', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))

                #grad_out += grad_temp
                #loss = loss - classifier_loss
                #self.metrics_accumulator.update_metric("classifier_loss", classifier_loss.item())

            if self.args.range_lambda != 0:

                r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                #loss = loss + r_loss
                self.metrics_accumulator.update_metric("range_loss", r_loss.item())

            if self.args.background_preservation_loss:


                if self.args.lp_custom:



                    """
                    print('norm t', t)
                    t = self.unscale_timestep(t)
                    print('norm t', t)
                    out = self.diffusion.p_mean_variance(
                        self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                    )

                    x_in = out["pred_xstart"]
                    if self.mask is not None:
                        masked_background = x_in  # * (1 - self.mask)
                    else:
                        masked_background = x_in
                    """

                    #lp_loss = ((masked_background - self.init_image).view(len(self.init_image), -1).norm(p=self.args.lp_custom, dim=1)**self.args.lp_custom).sum() #* self.args.l2_sim_lambda
                    #lp_loss = compute_lp_dist(masked_background - self.init_image, self.args.lp_custom).mean()
                    #lp_loss.backward(inputs=x)
                    #grad_temp = x.grad.data.clone()
                    #x.grad.data.zero_()
                    #grad_temp = torch.autograd.grad(lp_loss, x)[0].detach()
                    #grad_temp *= self.args.lp_custom_value
                    #grad_out -= grad_temp
                    #print('lp custom', self.args.lp_custom, self.args.lp_custom_value)
                    #print('x3', masked_background.norm(p=2))#, masked_background)


                    grad_temp = (x_in - self.init_image)
                    #lp_loss = ((masked_background - self.init_image).view(len(self.init_image), -1).norm(p=self.args.lp_custom, dim=1)**self.args.lp_custom).sum() * self.args.lp_custom_value
                    #grad_temp = (self.args.lp_custom * grad_temp.abs()**(self.args.lp_custom-1)) * grad_temp.sign()

                    if self.args.lp_custom < 1:
                        grad_temp = (self.args.lp_custom * (grad_temp.abs()+self.small_const)**(self.args.lp_custom-1)) * grad_temp.sign()
                    else:
                        grad_temp = (self.args.lp_custom * grad_temp.abs()**(self.args.lp_custom-1)) * grad_temp.sign()


                    #print('grad', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))
                    #norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                    #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                    #print('gradient lp custom norms before', norms_per_image)
                    #self.writer.add_scalars(
                    #    'Gradients of norm' + ', seed:' + str(
                    #        self.args.seed) + f', l{self.args.lp_custom} Regularization,class:/l{self.args.lp_custom}',
                    #    norms_per_image, self.tensorboard_counter)

                    if self.args.quantile_cut != 0:
                        #self.writer.add_images('images of norm before cut',
                        #                       min_max_scale(grad_temp.abs().sum(dim=1).unsqueeze(1)),
                        #                       self.tensorboard_counter)
                        bin_mask = grad_temp.abs().sum(dim=1).unsqueeze(1) <= grad_temp.abs().sum(
                            dim=1).view(grad_temp.shape[0], -1).quantile(self.args.quantile_cut, dim=1).view(-1,
                                                                                                                 1,
                                                                                                                 1,
                                                                                                                 1)
                        bin_mask_repeated_channels = torch.tile(bin_mask, dims=(1, 3, 1, 1))
                        grad_temp = torch.where(bin_mask_repeated_channels, grad_temp, grad_temp*0)
                        #self.writer.add_images('images of norm after cut',
                        #                       min_max_scale(grad_temp.abs().sum(dim=1).unsqueeze(1)),
                        #                       self.tensorboard_counter)

                    grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)

                    if self.args.enforce_same_norms:
                        # alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                        grad_temp *= self.args.lp_custom_value * eps.view(x.shape[0], -1).norm(p=2, dim=1).view(
                            x.shape[0], 1, 1, 1)

                        #norms_per_image = {str(i_val[0]): i_val[1].item() for i_val in
                        #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}

                        #self.writer.add_scalars(f'Gradient norms after normalization/l{self.args.lp_custom} norm', norms_per_image, self.tensorboard_counter)

                        #print(
                        #'enforce_same_norms: norms norms', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1),
                        #eps.view(x.shape[0], -1).norm(p=2, dim=1))
                    else:
                        grad_temp *= self.args.lp_custom_value
                    #print('norms classifier norms after', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))

                    grad_out -= grad_temp
                    #loss = (
                    #    loss
                    #    + lp_loss
                    #)
                    #loss = (
                    #        loss
                    #        + ((masked_background - self.init_image).view(len(self.init_image), -1).norm(p=1.5, dim=1)**1.5).mean() * self.args.l2_sim_lambda
                    #
                    #        #+ mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                    #)


            #with torch.enable_grad():
                #x = x.detach().requires_grad_()
                #t = self.unscale_timestep(t)
                #out = self.diffusion.p_mean_variance(
                #    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                #)
                #x_in = out["pred_xstart"] * fac + x * (1 - fac)
                #print("x_in_1", x_in.norm(p=2), x_in, fac)
                #x_in = out["pred_xstart"]
                #if self.mask is not None:
                #    masked_background = x_in  # * (1 - self.mask)
                #else:
                #    masked_background = x_in

                #self.writer.add_images('images for norms', masked_background.add(1).div(2).clamp(0, 1), self.tensorboard_counter)

                #print('x1', masked_background.norm(p=2), x_in.norm(p=2), x.norm(p=2))#, masked_background)

                if self.args.mssim_lambda:
                    #print('msssim min, max are', masked_background.min(), masked_background.max(), ((masked_background+1)/2).min(), ((masked_background+1)/2).max(), ((self.init_image+1)/2).min(), ((self.init_image+1)/2).max())

                    mssim_loss = 1 - ms_ssim( ((masked_background+1)/2).clamp(0, 1), (self.init_image+1)/2, data_range=1, size_average=False).sum()
                    grad_temp = torch.autograd.grad(mssim_loss, x, retain_graph=True)[0]
                    #norms_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                    #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                    #print('gradient mssim norms before', norms_per_image)
                    #self.writer.add_scalars(
                    #    'Gradients of norm' + ', seed:' + str(self.args.seed) + ', l1.5 Regularization,class:/mssim',
                    #    norms_per_image, self.tensorboard_counter)

                    grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)

                    if self.args.enforce_same_norms:
                        # alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                        grad_temp *= self.args.mssim_lambda * eps.view(x.shape[0], -1).norm(p=2, dim=1).view(
                            x.shape[0], 1, 1, 1)

                        #norms_per_image = {str(i_val[0]): i_val[1].item() for i_val in
                        #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}

                        #self.writer.add_scalars(f'Gradient norms after normalization/MSSIM norm', norms_per_image,
                        #                        self.tensorboard_counter)
                        #print('enforce_same_norms: norms lpips', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1),
                        #      eps.view(x.shape[0], -1).norm(p=2, dim=1))
                    else:
                        grad_temp *= self.args.mssim_lambda
                    #print('gradient lpips norms after', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))

                    #if not skip_backgrond:
                    grad_out -= grad_temp

                if self.args.ssim_lambda:
                    #print('sssim min, max are', masked_background.min(), masked_background.max(), ((masked_background+1)/2).min(), ((masked_background+1)/2).max(), ((self.init_image+1)/2).min(), ((self.init_image+1)/2).max())

                    ssim_loss = 1 - ssim( ((masked_background+1)/2).clamp(0, 1), (self.init_image+1)/2, data_range=1, size_average=False).sum()
                    grad_temp = torch.autograd.grad(ssim_loss, x)[0]
                    #norms_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                    #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                    #print('gradient mssim norms before', norms_per_image)
                    #self.writer.add_scalars(
                    #    'Gradients of norm' + ', seed:' + str(self.args.seed) + ', l1.5 Regularization,class:/ssim',
                    #    norms_per_image, self.tensorboard_counter)

                    grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)

                    if self.args.enforce_same_norms:
                        # alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                        grad_temp *= self.args.ssim_lambda * eps.view(x.shape[0], -1).norm(p=2, dim=1).view(
                            x.shape[0], 1, 1, 1)

                        #norms_per_image = {str(i_val[0]): i_val[1].item() for i_val in
                        #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}

                        #self.writer.add_scalars(f'Gradient norms after normalization/MSSIM norm', norms_per_image,
                        #                        self.tensorboard_counter)
                        #print('enforce_same_norms: norms lpips', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1),
                        #      eps.view(x.shape[0], -1).norm(p=2, dim=1))
                    else:
                        grad_temp *= self.args.ssim_lambda
                    #print('gradient lpips norms after', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))

                    #if not skip_backgrond:
                    grad_out -= grad_temp

                if self.args.lpips_sim_lambda:
                    lpips_loss = self.lpips_model(masked_background, self.init_image).sum()  # * self.args.lpips_sim_lambda
                    grad_temp = torch.autograd.grad(lpips_loss, x)[0]#.detach()
                    #print('x1_grad', masked_background.norm(p=2), x_in.norm(p=2), x.norm(p=2), masked_background)
                    #print('out1_grad', out["pred_xstart"].norm(p=2), out["pred_xstart"])
                    #norms_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                    #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                    #print('gradient lpips norms before', norms_per_image)
                    #self.writer.add_scalars(
                    #    'Gradients of norm' + ', seed:' + str(self.args.seed) + ', l1.5 Regularization,class:/lpips',
                    #    norms_per_image, self.tensorboard_counter)

                    grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)

                    if self.args.enforce_same_norms:
                        # alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                        grad_temp *= self.args.lpips_sim_lambda * eps.view(x.shape[0], -1).norm(p=2, dim=1).view(
                            x.shape[0], 1, 1, 1)

                        #norms_per_image = {str(i_val[0]): i_val[1].item() for i_val in
                        #                   enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}

                        #self.writer.add_scalars(f'Gradient norms after normalization/LPIPS norm', norms_per_image,
                        #                        self.tensorboard_counter)
                        #print('enforce_same_norms: norms lpips', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1),
                        #      eps.view(x.shape[0], -1).norm(p=2, dim=1))
                    else:
                        grad_temp *= self.args.lpips_sim_lambda
                    #print('gradient lpips norms after', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))

                    grad_out -= grad_temp
                    # loss = (
                    #        loss
                    #        + self.lpips_model(masked_background, self.init_image).sum()
                    #        * self.args.lpips_sim_lambda
                    # )

            #grad_out = torch.autograd.grad(loss, x)[0]
            #if self.verbose and False:
            #    titles = self.return_metrics_per_image(
            #        x.add(1).div(2).clamp(0, 1))
            #    assert len(self.init_probs) == len(self.probs) == len(titles)

            #    if self.args.projecting_cone and self.args.second_classifier_type != -1:
            #        if self.args.third_classifier_type != -1:
            #            titles = [f'p_i_rob;p_i;p_i_neg:{self.init_probs[i_x[0]]:.2f};{self.init_probs_second_classifier[i_x[0]]:.2f};{self.init_probs_third_classifier[i_x[0]]:.2f},\np_e_rob;p_e;p_i_neg:{self.probs[i_x[0]]:.2f};{self.probs_second_classifier[i_x[0]]:.2f};{self.probs_third_classifier[i_x[0]]:.2f},' + i_x[1].replace('L2', '\nL2')  + f'target:{self.text_batch[i_x[0]]}' for i_x in enumerate(titles)]
            #        else:
            #            titles = [f'p_i_rob;p_i:{self.init_probs[i_x[0]]:.2f};{self.init_probs_second_classifier[i_x[0]]:.2f},p_e_rob;p_e:{self.probs[i_x[0]]:.2f};{self.probs_second_classifier[i_x[0]]:.2f},' + i_x[1].replace('L2', '\nL2')  + f'target:{self.text_batch[i_x[0]]}' for i_x in enumerate(titles)]

            #    else:
            #        titles = [f'p_i:{self.init_probs[i_x[0]]:.2f},p_e:{self.probs[i_x[0]]:.2f},' + i_x[1].replace('L2', '\nL2')  + f'target:{self.text_batch[i_x[0]]}' for i_x in enumerate(titles)]
            #    self.writer.add_images('images for classifier', self.images_with_titles(
            #        x.add(1).div(2).clamp(0, 1), titles),
            #                           self.tensorboard_counter)

            #norms_per_image = {str(i_val[0]): i_val[1].item() for i_val in
            #                   enumerate(grad_out.view(x.shape[0], -1).norm(p=2, dim=1))}

            #self.writer.add_scalars('Total grad norm', norms_per_image, self.tensorboard_counter)


            #norms_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
            #                   enumerate(eps.view(x.shape[0], -1).norm(p=2,
            #                                                           dim=1))}
            #print('scorenet', norms_per_image)
            #self.writer.add_scalars('Scorenet norm', norms_per_image, self.tensorboard_counter)
            #print('gradient classifier norms before', norms_per_image)
            self.tensorboard_counter += 1

            #del grad_temp, x, masked_background, x_in, norms_per_image, titles, out, eps
            #print('cond fn time', time.time() - start_cond_fn)

            return grad_out #-grad_out #-torch.autograd.grad(loss, x)[0]

        def cond_fn_newer(x, t, y=None, eps=None):

            right_t = t.clone()
            print('right_t', right_t)
            print(right_t/1000.0, self.args.timestep_respacing)
            right_t = (right_t / 1000.0)*int(self.args.timestep_respacing)
            if self.args.prompt == "":
                return torch.zeros_like(x)


            # x_in = out["pred_xstart"]

            grad_out = torch.zeros_like(x)

            if self.args.clip_guidance_lambda != 0:
                with torch.enable_grad():
                    x = x.detach().requires_grad_()
                    t = self.unscale_timestep(t)

                    out = self.diffusion.p_mean_variance(
                        self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                    )
                    fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    print('using clip guidance')
                    clip_loss = self.clip_loss(x_in, text_embed)  # * self.args.clip_guidance_lambda
                    grad_temp = torch.autograd.grad(clip_loss, x)[0].detach()
                    grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)
                    grad_temp *= self.args.clip_guidance_lambda
                    grad_out -= grad_temp
                    # loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
            if self.args.classifier_lambda != 0:
                with torch.enable_grad():
                    x = x.detach().requires_grad_()
                    t = self.unscale_timestep(t)

                    out = self.diffusion.p_mean_variance(
                        self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                    )
                    pred_xstart_cache = out["pred_xstart"].clone()
                    fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                    fac_cached = fac.copy()

                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    loss_temp = torch.tensor(0.0).to(self.device)
                    if self.mask is not None:
                        masked_input = x_in * self.mask
                    else:
                        masked_input = x_in

                    logits = self.classifier(self.image_augmentations(masked_input).add(1).div(2).clamp(0, 1))

                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    # loss_indiv = log_probs[range(len(logits)), y.view(-1)]
                    loss_indiv = log_probs[
                        range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                    for i in range(current_bs):
                        # We want to average at the "augmentations level"
                        loss_temp += loss_indiv[i:: current_bs].mean()
                    print('shape loss', loss_indiv.shape)
                    print('targets', y.shape, y)
                    print('probs', logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)])
                    print('dist', (self.init_image - x_in).view(len(logits), -1).norm(p=2, dim=1))
                    self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]
                    probs_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in enumerate(self.probs)}

                    self.writer.add_scalars('Probs,classifier', probs_per_image, self.tensorboard_counter)

                    self.y = y
                    classifier_loss = loss_temp  # * self.args.classifier_lambda
                    grad_temp = torch.autograd.grad(classifier_loss, x)[0]#.detach()

                    norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                                       enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                    print('gradient classifier norms before', norms_per_image)
                    self.writer.add_scalars(
                        'Gradients of norm' + ', seed:' + str(self.args.seed) + ', l1.5 Regularization, class:/classifier', norms_per_image, self.tensorboard_counter)

                    grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)

                    if self.args.enforce_same_norms:
                        #alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                        grad_temp *= self.args.classifier_lambda*eps.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)
                        print('enforce_same_norms: norms classifier', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1), eps.view(x.shape[0], -1).norm(p=2, dim=1))
                    else:
                        grad_temp *= self.args.classifier_lambda
                    print('gradient classifier norms after', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))

                    grad_out += grad_temp
                    # loss = loss - classifier_loss
                    #self.metrics_accumulator.update_metric("classifier_loss", classifier_loss.item())
            # classifier_gradient = -torch.autograd.grad(loss, x, retain_graph=True)[0].detach()
            # with torch.enable_grad():
            #    x = x.detach().requires_grad_()

            if self.args.range_lambda != 0:
                with torch.enable_grad():
                    x = x.detach().requires_grad_()

                    #out = self.diffusion.p_mean_variance(
                    #    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                    #)

                    #fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                    out["pred_xstart"] = pred_xstart_cache.clone()
                    fac = fac_cached.copy()
                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    r_loss = range_loss(out["pred_xstart"]).sum()  # * self.args.range_lambda
                    grad_temp = torch.autograd.grad(r_loss, x)[0].detach()
                    grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1, 1)
                    grad_temp *= self.args.range_lambda
                    grad_out -= grad_temp
                    # loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

            print('skip background loss', (int(self.args.timestep_respacing) - self.args.skip_timesteps - self.tensorboard_counter < self.args.range_t), self.tensorboard_counter)
            if self.args.background_preservation_loss: #and (int(self.args.timestep_respacing) - self.args.skip_timesteps - self.tensorboard_counter >= self.args.range_t):

                    if self.args.lpips_sim_lambda:
                        with torch.enable_grad():
                            x = x.detach().requires_grad_()

                            # out = self.diffusion.p_mean_variance(
                            #    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                            # )

                            # fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                            out["pred_xstart"] = pred_xstart_cache.clone()
                            fac = fac_cached.copy()
                            x_in = out["pred_xstart"] * fac + x * (1 - fac)
                            if self.mask is not None:
                                print('using mask')
                                ##masked_background = x_in * (1 - self.mask)
                                masked_background = x_in * self.mask
                            else:
                                print('not using mask')
                                masked_background = x_in
                            # loss = (
                            #    loss
                            #    + self.lpips_model(masked_background, self.init_image).sum()
                            #    * self.args.lpips_sim_lambda
                            # )

                            grad_temp = torch.autograd.grad(self.lpips_model(masked_background, self.init_image).sum(), x)[
                                0].detach()
                            norms_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                                               enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                            print('gradient lpips norms before', norms_per_image)
                            self.writer.add_scalars(
                                'Gradients of norm' + ', seed:' + str(self.args.seed) + ', l1.5 Regularization,class:/lpips',
                                norms_per_image, self.tensorboard_counter)

                            grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1,
                                                                                           1)
                            if self.args.enforce_same_norms:
                                # alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                                grad_temp *= self.args.lpips_sim_lambda * eps.view(x.shape[0], -1).norm(p=2, dim=1).view(
                                    x.shape[0], 1, 1, 1)
                                print(
                                    'enforce_same_norms: norms regularizer',
                                    grad_temp.view(x.shape[0], -1).norm(p=2, dim=1),
                                    eps.view(x.shape[0], -1).norm(p=2, dim=1))
                            else:
                                grad_temp *= self.args.lpips_sim_lambda

                            grad_out -= grad_temp

                    if self.args.TV_lambda:
                        with torch.enable_grad():
                            x = x.detach().requires_grad_()

                            # out = self.diffusion.p_mean_variance(
                            #    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                            # )

                            # fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                            out["pred_xstart"] = pred_xstart_cache.clone()
                            fac = fac_cached.copy()
                            x_in = out["pred_xstart"] * fac + x * (1 - fac)
                            if self.mask is not None:
                                print('using mask')
                                ##masked_background = x_in * (1 - self.mask)
                                masked_background = x_in * self.mask
                            else:
                                print('not using mask')
                                masked_background = x_in

                            print('using TV sim', self.args.TV_lambda)
                            grad_temp = torch.autograd.grad((total_variation_loss(masked_background - self.init_image)),
                                                            x)[0].detach()
                            # grad_temp = torch.autograd.grad(((x - self.diffusion.q_sample(torch.tile(self.init_image, dims=(x.shape[0], 1, 1, 1)), t,
                            #                       torch.randn(*x.shape, device=self.device))).view(1,-1).norm(p=1.5,dim=1) ** 1.5).mean(), x)[0].detach()

                            norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                                               enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                            print('gradient TV norms before', norms_per_image)
                            self.writer.add_scalars(
                                'Gradients of norm' + ', seed:' + str(
                                    self.args.seed) + ', TV Regularization,class:/TV',
                                norms_per_image, self.tensorboard_counter)

                            # grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1,
                            #                                                                  1)
                            grad_temp *= self.args.TV_lambda
                            grad_out -= grad_temp
                            print('TV scaled loss', (
                                    total_variation_loss(masked_background - self.init_image) * self.args.l2_sim_lambda))

                    if self.args.lp_custom:

                        out["pred_xstart"] = pred_xstart_cache.clone()
                        fac = fac_cached.copy()
                        x_in = out["pred_xstart"] * fac + x * (1 - fac)
                        if self.mask is not None:
                            print('using mask')
                            ##masked_background = x_in * (1 - self.mask)
                            masked_background = x_in * self.mask
                        else:
                            print('not using mask')
                            masked_background = x_in
                        print('using lp sim', self.args.lp_custom, self.args.lp_custom_value)

                        # print('weighting by the gradient of classifier', classifier_gradient.min(), classifier_gradient.max())
                        # classifier_gradient = (classifier_gradient - classifier_gradient.min()) / (classifier_gradient.max() - classifier_gradient.min())
                        # classifier_gradient = 1 - classifier_gradient
                        # print('reweighting by the gradient of classifier', classifier_gradient.min(), classifier_gradient.max(), classifier_gradient.shape)

                        # print('shapes reweighting', classifier_gradient.shape, (masked_background - self.init_image).shape, ((masked_background - self.init_image)*classifier_gradient).shape)
                        # loss = (
                        #    loss
                        #    + ((masked_background - self.init_image).view(len(self.init_image), -1).norm(p=1.5, dim=1)**1.5).mean() * self.args.l2_sim_lambda
                        #    #+ mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                        # )
                        #grad_temp = torch.autograd.grad(((masked_background - self.init_image).view(x.shape[0],
                        #                                                                            -1).norm(p=self.args.lp_custom,
                        #                                                                                     dim=1)**self.args.lp_custom).mean(),
                        #                                x)[0].detach()

                        grad_temp = (masked_background - self.init_image)
                        if self.args.lp_custom < 1:
                            grad_temp = (self.args.lp_custom * (grad_temp.abs()+self.small_const)**(self.args.lp_custom-1)) * grad_temp.sign()
                        else:
                            grad_temp = (self.args.lp_custom * grad_temp.abs()**(self.args.lp_custom-1)) * grad_temp.sign()
                        # grad_temp = torch.autograd.grad(((x - self.diffusion.q_sample(torch.tile(self.init_image, dims=(x.shape[0], 1, 1, 1)), t,
                        #                       torch.randn(*x.shape, device=self.device))).view(1,-1).norm(p=1.5,dim=1) ** 1.5).mean(), x)[0].detach()

                        norms_per_image = {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                                           enumerate(grad_temp.view(x.shape[0], -1).norm(p=2, dim=1))}
                        print('gradient lp custom norms before', norms_per_image)
                        self.writer.add_scalars(
                            'Gradients of norm' + ', seed:' + str(
                                self.args.seed) + f', l{self.args.lp_custom} Regularization,class:/l{self.args.lp_custom}',
                            norms_per_image, self.tensorboard_counter)

                        grad_temp /= grad_temp.view(x.shape[0], -1).norm(p=2, dim=1).view(x.shape[0], 1, 1,
                                                                                          1)
                        if self.args.enforce_same_norms:
                            # alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x.shape)
                            grad_temp *= self.args.lp_custom_value * eps.view(x.shape[0], -1).norm(p=2, dim=1).view(
                                x.shape[0], 1, 1, 1)
                            print(
                            'enforce_same_norms: norms regularizer', grad_temp.view(x.shape[0], -1).norm(p=2, dim=1),
                            eps.view(x.shape[0], -1).norm(p=2, dim=1))
                        else:
                            grad_temp *= self.args.lp_custom_value

                        grad_out -= grad_temp
                        print(self.args.lp_custom,' scaled loss', (
                                (masked_background - self.init_image).view(x.shape[0], -1).norm(p=1.5,
                                                                                                dim=1) ** 1.5).mean() * self.args.l2_sim_lambda)
            try:
                norms_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                               enumerate((x_in - self.init_image).view(x.shape[0], -1).norm(p=1.5,
                                                                                            dim=1))}
                print('l1.5', len(self.init_image), x.shape[0], norms_per_image)
                self.writer.add_scalars(
                    'l1.5 distance',
                    norms_per_image, self.tensorboard_counter)

                norms_per_image = {str(str(self.imagenet_labels[y[i_val[0]]])): i_val[1].item() for i_val in
                                   enumerate(eps.view(x.shape[0], -1).norm(p=2,
                                                                           dim=1))}
                print('scorenet', norms_per_image)
                self.writer.add_scalars('scorenet norm', norms_per_image, self.tensorboard_counter)

                # print('mse scaled loss', mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda)
                print('total losss', grad_out.view(x.shape[0], -1).norm(p=2, dim=1))

            except Exception as err:
                print(str(err))



            self.tensorboard_counter += 1
            return grad_out  # -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):

            if self.mask is not None:
                skip_background = (int(
                    self.args.timestep_respacing) - self.args.skip_timesteps - self.tensorboard_counter < self.args.range_t)

                #print('postprocessing mask, skip', skip_background)
                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
                #background_stage_t = torch.tile(
                #    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                #)
                #print(out["sample"].shape, self.mask.shape)
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)

                #print('sample min, max before', out["sample"].abs().max(), out["sample"].min(), out["sample"].max(), out["pred_xstart"].min(), out["pred_xstart"].max(), range_loss(out["pred_xstart"]).sum() * self.args.range_lambda)
                #print('projecting', self.args.eps_project)#*out["sample"].abs().max())
                #out["sample"] = background_stage_t + project_perturbation(out["sample"] - background_stage_t, eps=30*out["sample"].abs().max(), p=2)
                if False:
                    print('t2', t)

                    #t = self.unscale_timestep(t)
                    #print('t2', t)
                    #print('projecting t', t-1)
                    self.writer.add_images('images before denoising', out["sample"].add(1).div(2).clamp(0, 1),
                                           self.tensorboard_counter)
                    print('before denoising norm', out["sample"].view(out["sample"].shape[0], -1).norm(p=2, dim=1))
                    out_ = self.diffusion.p_mean_variance(
                        self.model, out["sample"], t, clip_denoised=False, model_kwargs={"y": y}
                    )
                    x_in = out_["pred_xstart"]
                    if self.mask is not None:
                        masked_background = x_in  # * (1 - self.mask)
                    else:
                        masked_background = x_in

                    self.writer.add_images('images before projection', masked_background.add(1).div(2).clamp(0, 1),
                                               self.tensorboard_counter)
                    print('before projection norm', masked_background.view(out["sample"].shape[0], -1).norm(p=2, dim=1))

                    center = self.diffusion.q_sample(self.init_image, t, torch.randn(x.shape, device=self.device))
                    masked_background = self.init_image + project_perturbation(out["sample"] - self.init_image, eps=self.args.eps_project, p=2) #center + project_perturbation(out["sample"] - center, eps=self.args.eps_project, p=2) #*out["sample"].abs().max(), p=2)#self.init_image + project_perturbation(out["sample"] - self.init_image, eps=self.args.eps_project, p=2)#*out["sample"].abs().max(), p=2)
                    out["projected"] = masked_background


                    if self.args.second_classifier_type != -1:
                        print('using ensemble')
                        logits = (0.5*self.classifier(
                            self.image_augmentations(masked_background).add(1).div(2).clamp(0, 1)).softmax(
                            1) + 0.5*self.second_classifier(
                            self.image_augmentations(masked_background).add(1).div(2).clamp(0, 1)).softmax(1))
                    else:
                        #logits = self.classifier(self.image_augmentations(masked_background).add(1).div(2).clamp(0, 1))
                        logits = self.classifier(masked_background, t)


                    #logits = self.classifier(self.image_augmentations(masked_background).add(1).div(2))
                    #logits = self.classifier(masked_background.add(1).div(2))
                    #self.writer.add_images('images for classifier', masked_background.add(1).div(2).clamp(0, 1)[:self.args.batch_size], self.tensorboard_counter)
                    if self.args.second_classifier_type != -1:
                        log_probs = logits.log()
                    else:
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    # loss_indiv = log_probs[range(len(logits)), y.view(-1)]
                    loss_indiv = log_probs[
                        range(self.args.batch_size * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]

                    print(
                    'probs', logits[:self.args.batch_size].softmax(1)[range(self.args.batch_size), y.view(-1)])
                    print('dist', (self.init_image - x).view(len(logits), -1).norm(p=2, dim=1))

                    if self.args.second_classifier_type != -1:
                        probs_projected = logits[:self.args.batch_size][range(self.args.batch_size), y.view(-1)]
                    else:
                        probs_projected = logits[:self.args.batch_size].softmax(1)[range(self.args.batch_size), y.view(-1)]

                    fac = 0.01# * self.tensorboard_counter #self.diffusion.sqrt_one_minus_alphas_cumprod[0]
                    #bin_mask = (self.init_image-x).abs().sum(dim=1).unsqueeze(1) <= 0.5
                    #bin_mask_repeated_channels = torch.tile(bin_mask, dims=(1, 3, 1, 1))
                    out["sample"] = masked_background * fac + out["sample"] * (1 - fac) #torch.where(bin_mask_repeated_channels, masked_background, out["sample"]) #masked_background * fac + out["sample"] * (1 - fac)
                    print('factor is', fac)
                    titles = self.return_metrics_per_image(
                        masked_background.add(1).div(2).clamp(0, 1))
                    assert len(self.init_probs) == len(self.probs) == len(titles)

                    titles = [f'p_i:{self.init_probs[i_x[0]]:.2f},p_e:{probs_projected[i_x[0]]:.2f},' + i_x[1].replace('L2',
                                                                                                                  '\nL2')
                              for i_x in enumerate(titles)]

                    self.writer.add_images('images after projection', self.images_with_titles(
                        masked_background.add(1).div(2).clamp(0, 1), titles), self.tensorboard_counter)

                    self.writer.add_images('images after noising', out["sample"].add(1).div(2).clamp(0, 1), self.tensorboard_counter)

                #print('sample min, max after', out["sample"].abs().max(), out["sample"].min(), out["sample"].max(), out["pred_xstart"].min(), out["pred_xstart"].max(), range_loss(out["pred_xstart"]).sum() * self.args.range_lambda)
            return out

        save_image_interval = self.diffusion.num_timesteps // 5
        targets_classifier = y #[self.args.target_class]  # [293]*4 #[979]*4 #[293, 293, 293, 293] #[286, 287, 293, 288] #[979, 973, 980, 974] [286, 287, 293, 288]

        if self.args.gen_type == 'ddim':
            gen_func = self.diffusion.ddim_sample_loop_progressive
        elif self.args.gen_type == 'p_sample':
            gen_func = self.diffusion.p_sample_loop_progressive
        else:
            raise ValueError(f'Generation type {self.args.gen_type} is not implemented.')

        samples = gen_func(#ddim_sample_loop_progressive(#p_sample_loop_progressive(#ddim_sample_loop_progressive(#p_sample_loop_progressive(
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
                # torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
            },
            cond_fn=cond_fn_blended if self.args.use_blended else cond_fn_clean, #cond_fn,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=self.init_image,
            postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
            randomize_class=False,
            resizers=self.resizers,
            range_t=self.args.range_t,
            eps_project=self.args.eps_project,
            ilvr_multi=self.args.ilvr_multi,
            seed=self.args.seed
            #mask=self.mask

        )

        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
        print('num total steps is', total_steps)
        for i, sample in enumerate(samples):
            print(i)
            if i == total_steps:
                #print('saving at sample', i)
                sample_final = sample

        self.writer.flush()
        self.writer.close()
        return sample_final["pred_xstart"].add(1).div(2).clamp(0, 1)

    def perturb(self, x, y, dir):

        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        self.tensorboard_counter = 0
        adv_best = self.edit_image_by_prompt(x, y, dir)
        return [adv_best]
