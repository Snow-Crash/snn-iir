# -*- coding: utf-8 -*-

"""
# File Name : utilities.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: utility functions.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import matplotlib
import torch
from torch.utils.data import Dataset, DataLoader

# matplotlib.use('Qt5Agg')

def generate_rand_pattern(pattern_num, synapse_num, length, min_spike_num, max_spike_num):
    """
        Create random test case. Each pattern belongs to different class
        Each test case has multiple spike trains, corresponding to different synapse.
        1 indicates a spike, 0 indicates no spike.
        pattern_num: number of random patterns
        synapse_num: number of spike trains of each pattern
        length: length of patterns
        min_spike_num: minimum number of spikes in each spike train
        max_spike_num: maximum number of spikes in each spike train
            if min_spike_num == max_spike_num, all spike trains have same number of spikes
        x_train: [pattern_idx, synapse_num, time]
        y_train_onehot: [pattern_num, pattern_num], one hot label
        y_train_cat: [pattern_number], categorical label

    """
    x_train = np.zeros([pattern_num, synapse_num, length], dtype=np.float32)
    y_train_onehot = np.zeros([pattern_num, pattern_num], dtype=np.float32)
    y_train_cat = np.zeros(pattern_num, dtype=np.float32)

    for i in range(pattern_num):
        for j in range(synapse_num):
            spike_number = random.randint(min_spike_num, max_spike_num)
            spike_time = random.sample(range(length), spike_number)
            x_train[i, j, spike_time] = 1
        y_train_onehot[i, i] = 1
        y_train_cat[i] = i

    return x_train, y_train_onehot, y_train_cat


def filter_spike(spike_train, filter_type='exp', tau_m=10, tau_s=2.5,
                 normalize=True):
    """
        generate filtered spike train
        spike_train: 1d array, 1 represents spike
        filter_type: exp or dual_exp
        tau_m: time constant used by dual_exp
        tau_s: time constant used by exp and dual exp
    """
    length = len(spike_train)
    eta = tau_m / tau_s
    v_0 = np.power(eta, eta / (eta - 1)) / (eta - 1)

    psp_m = 0
    psp_s = 0
    target_pattern = np.zeros([1, length], dtype=np.float32)
    if filter_type == 'dual_exp':
        for i in range(length):
            psp_m = psp_m * np.exp(-1 / tau_m) + spike_train[i]
            psp_s = psp_s * np.exp(-1 / tau_s) + spike_train[i]
            if normalize:
                target_pattern[0, i] = (psp_m - psp_s) * v_0
            else:
                target_pattern[0, i] = (psp_m - psp_s)
    elif filter_type == 'exp':
        for i in range(length):
            psp_s = psp_s * np.exp(-1 / tau_s) + spike_train[i]
            target_pattern[0, i] = psp_s

    return target_pattern


def filter_spike_multiple(spike_trains, filter_type='exp', tau_m=10, tau_s=2.5,
                          normalize=True):
    """
    create filtered spike train for a batch

    spike_train_batch[number of spike_trains, time]
    """

    spike_train_num, time = spike_trains.shape
    filtered_spikes = np.zeros(spike_trains.shape, dtype=np.float32)

    # for each spike train in the instance
    for i in range(spike_train_num):
        filtered_spikes[i] = filter_spike(spike_trains[i], filter_type=filter_type,
                                          tau_m=tau_m,tau_s=tau_s, normalize=normalize)

    return filtered_spikes

def mutate_spike_pattern(template_pattern, mean, sigma):
    """
    create new spike pattern based on provided template, jitter follows normal distribution
    :param template_pattern: 2d array[input_dimension, time]
    :param mean: mean of normal distribution
    :param sigma: standard deviation of normal distribution
    :return: 2d array [input_dimension, time]
    """

    input_size, length = template_pattern.shape
    mutated_pattern = np.zeros([input_size, length],dtype=np.float32)
    input_idx, spike_time = np.where(template_pattern != 0)
    delta_t = np.rint(np.random.normal(mean, sigma, spike_time.shape)).astype(int)
    mutated_spike_time = spike_time + delta_t

    # print(delta_t)

    # find the time larger than time range, set to maximum time
    mutated_spike_time[np.where(mutated_spike_time >= length)] = length - 1
    # find the time less than 0, set to 0
    mutated_spike_time[np.where(mutated_spike_time < 0)] = 0

    mutated_pattern[input_idx, mutated_spike_time] = 1

    return mutated_pattern

def plot_raster(spike_mat, **kwargs):
    """
    spike_mat[row, time]
    """
    neuron_idx, spike_time = np.where(spike_mat != 0)
    # plt.figure()
    plt.plot(spike_time, neuron_idx, linestyle='None', marker='|', **kwargs)

    #    print(**kwargs)
    if 'label' in kwargs:
        plt.legend(loc='upper right', fontsize='x-large')

    # plt.show()

def plot_raster_dot(spike_mat):
    '''
    another function to plot spikes
    :param spike_mat: [row, length/time]
    :return:
    '''
    h,w = spike_mat.shape
    plt.figure()
    point_coordinate = np.where(spike_mat != 0)
    plt.scatter(point_coordinate[1], point_coordinate[0], s=1.5)
    plt.gca().invert_yaxis()
    plt.gca().set_xlim([0, w])
    plt.gca().set_ylim([0, h])

def gaussian_filter_spike_train(spike_train, sigma):
    """
    create a spike probability over time
    :param spike_train: 1d array[time]
    :param sigma:
    :return: spike probability, 1d array[time]
    """

    spike_probability = filters.gaussian_filter(spike_train, sigma, mode='constant', cval=0)

    return spike_probability.astype(np.float32)


def gaussian_filter_spike_train_batch(spike_train_batch, sigma):
    """

    :param spike_trains: 3d array [pattern_id, spike_train_id, time]
    :param sigma:
    :return:
    """

    batch_size, spike_train_num, time = spike_train_batch.shape
    filtered_spike_batch = np.zeros(spike_train_batch.shape, dtype=np.float32)

    for i in range(batch_size):
        for j in range(spike_train_num):
            filtered_spike_batch[i, j] = gaussian_filter_spike_train(spike_train_batch[i, j], sigma)

    return filtered_spike_batch

class RandPatternDataset(Dataset):
    """random pattern dataser"""

    def __init__(self, dataset_path, label_path, transform=None):
        self.dataset = np.load(dataset_path)
        self.dataset = self.dataset.astype(np.float32)
        self.label = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.dataset[idx],self.label[idx]

class monitor():

    def __init__(self, snn_model, batch_size, length):
        '''

        :param snn_model:
        :param batch_size:
        :param length:
        '''
        self.v = torch.zeros([batch_size, snn_model.neuron_number, length])
        self.spike = torch.zeros([batch_size, snn_model.neuron_number, length])
        self.filtered_spike = torch.zeros([batch_size, snn_model.neuron_number, length])
        self.reset_v = torch.zeros([batch_size, snn_model.neuron_number, length])

        self.v_0 = snn_model.v_0

        self.step_counter = 0

    def record_dict(self, spike, states):

        self.spike[:, :, self.step_counter] = spike
        self.filtered_spike[:, :, self.step_counter] = (states["filter_m"] - states["filter_s"]) * self.v_0
        self.v[:, :, self.step_counter] = states["v"]
        self.reset_v[:, :, self.step_counter] = states["reset_v"]

        self.step_counter += 1

    def record(self, spike, v, reset_v, filter_m, filter_s):
        self.spike[:, :, self.step_counter] = spike
        self.filtered_spike[:, :, self.step_counter] = (filter_m-filter_s) * self.v_0
        self.v[:, :, self.step_counter] = v
        self.reset_v[:, :, self.step_counter] = reset_v

        self.step_counter += 1


def float_to_spike_train(value, spike_train_length):
    '''
    convert a floating value to a spike train
    :param value: a floating value in [0,1.0]
    :param spike_train_length: length of spike train
    :return: spike_train: [spike_train_length]
    '''
    spike_train = np.zeros(spike_train_length)
    spike_number = int(value*spike_train_length)
    ticks = np.linspace(0,spike_train_length,num = spike_number, endpoint=False, dtype=np.int)
    spike_train[ticks] = 1

    return  spike_train

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    template_num = 10
    synapse_num = 40
    length = 200
    # test generate_rand_pattern
    spike_train_template, labels_onehot, labels_cat = generate_rand_pattern(10, 40, 200, 5, 10)

    # mutate spike pattern
    new_patterns = mutate_spike_pattern(spike_train_template[0], 0, 0.5)

    #plot new pattern and template pattern to see if they are similar
    plot_raster(new_patterns)
    plot_raster(spike_train_template[0])

    # for each spike train template, mutate it to create 100 spike trains
    mutate_num = 100
    # test_cases = np.zeros(mutate_num*template_num, synapse_num, length)
    # test_cases_label_onehot = np.zeros([mutate_num*template_num,template_num])
    # test_cases_label_cat = np.zeros(mutate_num * template_num)

    test_cases = []
    test_cases_label_onehot = []
    test_cases_label_cat = []
    filtered_target = []

    for template_idx, template in enumerate(spike_train_template):
        for j in range(mutate_num):
            test_cases.append(mutate_spike_pattern(template, 0, 0.5))
            test_cases_label_onehot.append(labels_onehot[template_idx])
            test_cases_label_cat.append(labels_cat[template_idx])
            target = np.zeros([template_num, length])
            target[template_idx, 10+template_idx*18] = 1

            filtered_target.append(filter_spike_multiple(target, filter_type='dual_exp', tau_m=10, tau_s=2.5))

    test_cases = np.stack(test_cases)
    test_cases_label_cat = np.stack(test_cases_label_cat)
    test_cases_label_onehot = np.stack(test_cases_label_onehot)
    filtered_target = np.stack(filtered_target)


    np.save("test_cases.npy", test_cases)
    np.save("test_case_label_onehot", test_cases_label_onehot)
    np.save("test_case_label_cat", test_cases_label_cat)
    np.save("filtered_target", filtered_target)

    for i in range(10):
        plt.plot(filtered_target[mutate_num,i])

    plt.show()







