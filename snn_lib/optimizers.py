# -*- coding: utf-8 -*-

"""
# File Name : optimizers.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: optimizers.
"""

import torch
import omegaconf
from omegaconf import OmegaConf

def get_optimizer(params, conf):
    optimizer_conf = conf['optimizer']

    optimizer_choice = optimizer_conf['optimizer_choice']

    if optimizer_choice == 'Adam':
        lr = optimizer_conf['Adam']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.Adam(params, lr)
    elif optimizer_choice == 'AdamW':
        lr = optimizer_conf['AdamW']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.AdamW(params, lr)
    elif optimizer_choice == 'SGD':
        lr = lr = optimizer_conf['SGD']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.SGD(params, lr)
    else:
        raise Exception('optimizer', optimizer_conf['optimizer_choice'] ,'not implemented.')
    
    