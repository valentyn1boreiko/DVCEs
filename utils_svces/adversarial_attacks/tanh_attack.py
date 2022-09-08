import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim
from .restartattack import RestartAttack
from .utils import project_perturbation, normalize_perturbation, create_early_stopping_mask

####################################################################################################
def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def img_to_tanh(x, boxmin=0, boxmax=1):
    boxmul = 0.5 * (boxmax - boxmin)
    boxplus = 0.5 * (boxmin + boxmax)
    return atanh((x - boxplus) / boxmul)

def tanh_to_img(w, boxmin=0, boxmax=1):
    boxmul = 0.5 * (boxmax - boxmin)
    boxplus = 0.5 * (boxmin + boxmax)
    #transform tanh space image to normal image space with bound [low_b, up_b]
    return torch.tanh(w) * boxmul + boxplus

def CW_loss(pert_img, orig_img, out, y_oh, targeted, reg_weight,confidence=0, reduce=True):
    bs = pert_img.shape[0]
    loss_1 = torch.sum(((pert_img - orig_img) ** 2).view(bs, -1), dim=1)

    # logits of gt class
    out_real = torch.sum((out * y_oh), 1)
    # logits of other highest scoring
    out_other = torch.max(out * (1. - y_oh) - y_oh * 100000000., 1)[0]

    if targeted:
        # maximize target class and minimize second highest
        loss_2 = torch.clamp_min(out_other - out_real, -confidence)
    else:
        # minimize target and max second highest
        loss_2 = torch.clamp_min(out_real - out_other, -confidence)

    if reduce:
        loss = torch.mean(reg_weight * loss_1 + loss_2)

    return loss

#https://arxiv.org/pdf/1608.04644.pdf
#Tensorflow: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
class TanhIterativeAttack(RestartAttack):
    def __init__(self, iterations, stepsize, num_classes, loss='CW', restarts=0, init_noise_generator=None, confidence=0.0, early_stopping=0, model=None, reg_weight=1):
        super().__init__(restarts=restarts, model=model, save_trajectory=False)
        self.iterations = iterations
        self.stepsize = stepsize
        self.num_classes = num_classes
        self.loss = loss
        self.init_noise_generator = init_noise_generator
        self.confidence = confidence
        self.reg_weight = reg_weight
        self.early_stopping=early_stopping

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'TanhIterative'
        dict['iterations'] = self.iterations
        dict['stepsize'] = self.stepsize
        dict['reg weight'] = self.reg_weight
        dict['confidence'] = self.confidence
        #config_dict['init_sigma'] = self.init_sigma
        return dict

    def perturb_inner(self, x, y, targeted=False, x_init=None):
        bs = y.shape[0]

        if self.loss == 'CW':
            y_oh = torch.nn.functional.one_hot(y, self.num_classes)
            y_oh = y_oh.float()
            l_f = lambda data, data_out: CW_loss(data, x, data_out, y_oh, targeted, self.reg_weight, confidence=self.confidence)
        else:
            l_f = lambda data, data_out: self.loss(data, data_out, x, y)


        data = x.clone().detach()


        data_tanh = img_to_tanh(data)

        if self.init_noise_generator is None:
            pert_tanh = torch.zeros_like(data)
        else:
            raise NotImplementedError()

        pert_tanh.requires_grad_(True)

        optimizer = optim.Adam([pert_tanh], self.stepsize)

        for i in range(self.iterations):
            optimizer.zero_grad()

            with torch.enable_grad():
                #distance to original image
                pert_img = tanh_to_img(data_tanh + pert_tanh)
                out = self.model(pert_img)
                loss = l_f(pert_img, out)

            if self.early_stopping > 0:
                conf, pred = torch.max(torch.nn.functional.softmax(out, dim=1), 1)
                conf_mask = conf > self.early_stopping
                if targeted:
                    correct_mask = torch.eq(y, pred)
                else:
                    correct_mask = (~torch.eq(y, pred))
                mask = (conf_mask & correct_mask).detach()
                saved_perts = pert_tanh[mask,:].detach()

                if sum(mask.float()) == x.shape[0]:
                    break

            loss.backward()
            optimizer.step()

            if self.early_stopping > 0:
                pert_tanh[mask] = saved_perts



        pert_img = tanh_to_img(data_tanh + pert_tanh)
        loss = l_f(pert_img, self.model(pert_img))
        return pert_img, loss, None

