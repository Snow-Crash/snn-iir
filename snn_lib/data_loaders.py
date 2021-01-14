# -*- coding: utf-8 -*-

"""
# File Name : data_loaders.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: data loaders.
"""

from torch.utils.data import Dataset, DataLoader

import numpy as np
from torchvision import transforms, utils
import torch

class MNISTDataset_Poisson_Spike(Dataset):
    """mnist dataset

    torchvision_mnist: dataset object
    length: number of steps of snn
    max_rate: a scale factor. MNIST pixel value is normalized to [0,1], and them multiply with this value
    faltten: return 28x28 image or a flattened 1d vector
    transform: transform
    """

    def __init__(self, torchvision_mnist, length, max_rate=1, flatten=False, transform=None):
        self.dataset = torchvision_mnist
        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)

        #shape of image [h,w]
        img = np.array(self.dataset[idx][0], dtype=np.float32) / 255.0 * self.max_rate
        shape = img.shape

        #flatten image
        img = img.reshape(-1)

        # shape of spike_trains [h*w, length]
        spike_trains = np.zeros((len(img), self.length), dtype=np.float32)

        #extend last dimension for time, repeat image along the last dimension
        img_tile = np.expand_dims(img,1)
        img_tile = np.tile(img_tile, (1,self.length))
        rand = np.random.uniform(0,1,(len(img), self.length))
        spike_trains[np.where(img_tile > rand)] = 1

        if self.flatten == False:
            spike_trains = spike_trains.reshape([shape[0], shape[1], self.length])

        return spike_trains, self.dataset[idx][1]

class MNISTDataset(Dataset):
    """mnist dataset

    torchvision_mnist: dataset object
    length: number of steps of snn
    max_rate: a scale factor. MNIST pixel value is normalized to [0,1], and them multiply with this value
    flatten: return 28x28 image or a flattened 1d vector
    transform: transform
    """

    def __init__(self, torchvision_mnist, length, max_rate = 1, flatten = False, transform=None):
        self.dataset = torchvision_mnist
        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)

        img = np.array(self.dataset[idx][0],dtype=np.float32)/255.0 * self.max_rate
        shape = img.shape
        img_spike = None
        if self.flatten == True:
            img = img.reshape(-1)

        return img, self.dataset[idx][1]

def get_rand_transform(transform_config):
    t1_size = transform_config['RandomResizedCrop']['size']
    t1_scale = transform_config['RandomResizedCrop']['scale']
    t1_ratio = transform_config['RandomResizedCrop']['ratio']
    t1 = transforms.RandomResizedCrop(t1_size, scale=t1_scale, ratio=t1_ratio, interpolation=2)
    
    t2_angle = transform_config['RandomRotation']['angle']
    t2 = transforms.RandomRotation(t2_angle, resample=False, expand=False, center=None)
    t3 = transforms.Compose([t1, t2])

    rand_transform = transforms.RandomApply([t1, t2, t3], p=transform_config['RandomApply']['probability'])

    return rand_transform