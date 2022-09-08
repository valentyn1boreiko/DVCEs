import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .adversarialattack import AdversarialAttack

class RestartAttack(AdversarialAttack):
    #Base class for attacks that start from different initial values
    #Make sure that they MINIMIZE the given loss function
    def __init__(self, loss, restarts,  num_classes, model=None, save_trajectory=False):
        super().__init__(loss, num_classes, model=model, save_trajectory=save_trajectory)
        self.restarts = restarts

    def perturb_inner(self, x, y, targeted=False, x_init=None):
        #force child class implementation
        raise NotImplementedError()

    def perturb(self, x, y, targeted=False, x_init=None):
        #base class method that handles various restarts
        self._check_model()

        is_train = self.model.training
        self.model.eval()

        restarts_data = x.new_empty((1 + self.restarts,) + x.shape)
        restarts_objs = x.new_empty((1 + self.restarts, x.shape[0]))

        if self.save_trajectory:
            self.last_trajectory = None
            trajectories_shape = (1 + self.restarts, self._get_trajectory_depth(),) + x.shape
            restart_trajectories = x.new_empty(trajectories_shape, device=torch.device('cpu'))

        for k in range(1 + self.restarts):
            k_data, k_obj, k_trajectory = self.perturb_inner(x, y, targeted=targeted, x_init=x_init)
            restarts_data[k, :] = k_data
            restarts_objs[k, :] = k_obj
            if self.save_trajectory:
                restart_trajectories[k, :] = k_trajectory.cpu()

        bs = x.shape[0]
        best_idx = torch.argmin(restarts_objs, 0)
        best_data = restarts_data[best_idx, range(bs), :]

        if self.save_trajectory:
            self.last_trajectory = restart_trajectories[best_idx, :, range(bs), :]

        #reset density_model status
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        return best_data
