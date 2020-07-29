# -*- coding: utf-8 -*-

"""
# File Name : schedulers.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: schedulers.
"""

import torch
import omegaconf
from omegaconf import OmegaConf


def get_scheduler(optimizer, conf):
    scheduler_conf = conf['scheduler']
    scheduler_choice = scheduler_conf['scheduler_choice']

    if scheduler_choice == 'MultiStepLR':
        milesones = list(scheduler_conf[scheduler_choice]['milestones'])
        print('scheduler:', scheduler_choice, 'milesones:', milesones)
        if 'gamma' in scheduler_conf[scheduler_choice]:
            gamma = scheduler_conf[scheduler_choice]['gamma']
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milesones, gamma)
        else:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milesones)

    elif scheduler_choice == 'CosineAnnealingWarmRestarts':
        T_0 = scheduler_conf[scheduler_choice]['T_0']
        print('scheduler:', scheduler_choice, 'T_0:', T_0)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)

    elif scheduler_choice == 'CyclicLR':
        base_lr = scheduler_conf[scheduler_choice]['base_lr']
        max_lr = scheduler_conf[scheduler_choice]['max_lr']
        step_size_up = scheduler_conf[scheduler_choice]['step_size_up']
        print('scheduler:', scheduler_conf['scheduler_choice'], 'base_lr:', base_lr, 
        'max_lr:', max_lr, 'step_size_up:', step_size_up)
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up)
    
    elif scheduler_choice == 'none':
        return None
    else:
        raise Exception('scheduler', scheduler_conf['scheduler_choice'] ,'not implemented.')