# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        ctx.training = training
        ctx.p_drop = p_drop
        if training:
            gate = torch.empty(1, device=x.device).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.empty(x.size(0), device=x.device).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        training = ctx.training
        p_drop = ctx.p_drop
        if training:
            gate = ctx.saved_tensors[0]
            if gate.item() == 0:
                beta = torch.empty(grad_output.size(0), device=grad_output.device).uniform_(0, 1)
                beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
                beta = Variable(beta)
                return beta * grad_output, None, None, None
            else:
                return grad_output, None, None, None
        else:
            return (1 - p_drop) * grad_output, None, None, None


class ShakeDrop(nn.Module):

    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)
