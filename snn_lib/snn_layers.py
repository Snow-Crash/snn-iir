# -*- coding: utf-8 -*-

"""
# File Name : snn_layers.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: spiking neural network layers.
"""

import torch
import numpy as np
import time

class threshold(torch.autograd.Function):
    """
    heaviside step threshold function
    """

    @staticmethod
    def forward(ctx, input, sigma):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        """

        """
        ctx.save_for_backward(input)
        ctx.sigma = sigma

        output = input.clone()
        output = torch.max(torch.tensor(0.0,device=output.device),torch.sign(output-1.0))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """

        """
        input, = ctx.saved_tensors
        sigma = ctx.sigma

        exponent = -torch.pow((1-input), 2)/(2.0*sigma**2)
        exp = torch.exp(exponent)

        erfc_grad = exp / (2.506628 * sigma)
        grad = erfc_grad * grad_output

        return grad, None

class threshold_layer(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

    def forward(self, input):
        return threshold.apply(input, self.sigma)


class filter_layer(torch.nn.Module):
    '''
    implement dual exponential filter.
    It is used to filter the output of last layer for temporal pattern training.
    '''
    def __init__(self, input_size, step_num, batch_size, filter_tau_m, filter_tau_s, device=None):

        super().__init__()
        self.input_size = input_size
        self.step_num = step_num
        self.filter_tau_s = torch.tensor(filter_tau_s)
        self.filter_tau_m = torch.tensor(filter_tau_m)

        filter_eta = torch.tensor(filter_tau_m / filter_tau_s)
        self.filter_v0 = torch.nn.Parameter(torch.pow(filter_eta, filter_eta / (filter_eta - 1)) / (filter_eta - 1))
        self.filter_v0.requires_grad = False

        self.filter_decay_m = torch.nn.Parameter(torch.exp(torch.tensor(-1/filter_tau_m)))
        self.filter_decay_m.requires_grad = False
        self.filter_decay_s = torch.nn.Parameter(torch.exp(torch.tensor(-1/filter_tau_s)))
        self.filter_decay_s.requires_grad = False

    def forward(self, input_spikes):
        # type: (Tensor) -> Tensor
        """
        Accept 3d tensot as input, which is a sequence of spikes of every time step
        :param input: [batch, neuron, t]
        :return: [batch, input_size, t]
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        # initial state of filter
        filter_state_s = torch.full((self.input_size,), 0.0).to(self.filter_v0.device)
        filter_state_m = torch.full((self.input_size,), 0.0).to(self.filter_v0.device)
        filter_output_list = []

        for t in range(len(inputs)):
            filter_state_m = filter_state_m * self.filter_decay_m + inputs[t]
            filter_state_s = filter_state_s * self.filter_decay_s + inputs[t]
            filtered_output = (filter_state_m - filter_state_s) * self.filter_v0

            filter_output_list += [filtered_output]

        # stack along last dimension
        return torch.stack(filter_output_list,dim=-1)

class exponential_filter_layer(torch.nn.Module):
    '''
    implement exponential exponential filter.
    It is used to filter the output of last layer for temporal pattern training.
    '''
    def __init__(self, input_size, step_num, batch_size, alpha, device):

        super().__init__()
        self.input_size = input_size
        self.step_num = step_num
        self.alpha = alpha

        self.device = device

    def forward(self, input_spikes):
        # type: (Tensor) -> Tensor
        """
        Accept 3d tensot as input, which is a sequence of spikes of every time step
        :param input: [batch, neuron, t]
        :return: [batch, input_size, t]
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        # initial state of filter
        filter_output_list = []

        filter_state = torch.full((self.input_size,), 0.0).to(self.device)

        for t in range(len(inputs)):
            filter_state = filter_state * (1 - self.alpha) + self.alpha*inputs[t]

            filter_output_list += [filter_state]

        # stack along last dimension
        return torch.stack(filter_output_list,dim=-1)


class double_exponential_filter_layer(torch.nn.Module):
    '''
    implement exponential exponential filter.
    It is used to filter the output of last layer for temporal pattern training.
    '''
    def __init__(self, input_size, step_num, batch_size, alpha, beta, device):

        super().__init__()
        self.input_size = input_size
        self.step_num = step_num
        self.alpha = alpha
        self.beta = beta

        self.device = device

    def forward(self, input_spikes):
        # type: (Tensor) -> Tensor
        """
        Accept 3d tensot as input, which is a sequence of spikes of every time step
        :param input: [batch, neuron, t]
        :return: [batch, input_size, t]
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        # initial state of filter
        filter_output_list = []

        s = torch.full((self.input_size,), 0.0).to(self.device)
        b = torch.full((self.input_size,), 0.0).to(self.device)

        s_t_minus_1 = torch.full((self.input_size,), 0.0).to(self.device)
        b_t_minus_1 = torch.full((self.input_size,), 0.0).to(self.device)

        for t in range(len(inputs)):

            if t == 0:
                pass
            elif t == 1:
                s = inputs[t]
                b = inputs[t] - inputs[t-1]
            else:
                s = self.alpha * inputs[t] + (1-self.alpha)(s_t_minus_1 + b_t_minus_1)

                b = self.beta* (s - s_t_minus_1) + (1 - self.beta) * b_t_minus_1

                s_t_minus_1 = s
                b_t_minus_1 = b

            filter_output_list += [s]

        # stack along last dimension
        return torch.stack(filter_output_list,dim=-1)

class snn_cell(torch.nn.Module):
    '''
    Implement the functionalty of spiking neuron.
    It times the psp with weight and calculate voltage.
    Forward function only calculate one step
    '''
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m, tau_s, train_tau_m,
                 train_tau_s):
        super().__init__()
        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size

        self.weight = torch.nn.Linear(input_size, neuron_number, bias=True)
        self.tau_m = torch.full((input_size,), tau_m)
        self.tau_s = torch.full((input_size,), tau_s)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full((self.neuron_number,),self.reset_decay))
        self.reset_decay.requires_grad = False

        self.reset_v = torch.nn.Parameter(torch.full((self.neuron_number,), 1.0))
        self.reset_v.requires_grad = False

        # calculate the norm factor to make max spike response to be 1
        eta = torch.tensor(tau_m / tau_s)
        self.v_0 = torch.nn.Parameter(torch.pow(eta, eta / (eta - 1)) / (eta - 1))
        self.v_0.requires_grad = False

        self.decay_m = torch.nn.Parameter(torch.tensor(torch.exp(-1 / self.tau_m)))
        self.decay_m.requires_grad = train_tau_m

        self.decay_s = torch.nn.Parameter(torch.tensor(torch.exp(-1 / self.tau_s)))
        self.decay_s.requires_grad = train_tau_s

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full((self.neuron_number,),self.decay_v))
        self.decay_v.requires_grad = False

    def forward(self, current_spike, prev_states):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        """
        :param current_spike: [batch, synapse_id]
        :param prev_states: (prev_v, prev_reset, prev_psp_m, prev_psp_s) tuple stores previous states
                prev_v[batch, neuron_number]
                prev_reset[batch, neuron_number]
                prev_psp_m[batch, input_size]
                prev_psp_s[batch, input_size]
        :return: spike[batch, neuron_num, t], new_states(current_v, current_reset, current_psp_m, current_psp_s)
        """

        # previous states
        prev_v, prev_reset, prev_psp_m, prev_psp_s = prev_states

        current_psp_m = prev_psp_m * self.decay_m + current_spike
        current_psp_s = prev_psp_s * self.decay_s + current_spike
        current_psp = current_psp_m - current_psp_s

        weighted_psp = self.weight(current_psp) * self.v_0

        # current_v = prev_v * self.decay_v + weighted_psp - prev_reset
        current_v = weighted_psp - prev_reset

        threshold_function = threshold.apply
        spike = threshold_function(current_v, self.sigma)

        current_reset = prev_reset * self.reset_decay + spike * self.reset_v

        new_states = (current_v, current_reset, current_psp_m, current_psp_s)

        return spike, new_states

class snn_layer(torch.nn.Module):
    '''
    wrapper of snn_cell. loop unroll implemented in forward function
    '''
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m,
                 tau_s, train_tau_m, train_tau_s):
        super().__init__()

        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size

        self.snn_cell = snn_cell(input_size, neuron_number, step_num, batch_size, tau_m, tau_s, train_tau_m,
                 train_tau_s)


    def forward(self, input_spikes, states):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        """
        :param input_spikes: [batch, input_size, t]
        :param states: (prev_v, prev_reset, prev_psp_m, prev_psp_s) tuple stores previous states
                prev_v[batch, neuron_number]
                prev_reset[batch, neuron_number]
                prev_psp_m[batch, input_size]
                prev_psp_s[batch, input_size]
        :return: spikes[neuron_number, length], [states0, state1...]
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        # length = len(input_spikes)
        spikes = []

        #iterate over each time step
        for i in range(len(inputs)):
            spike, states = self.snn_cell(inputs[i], states)
            spikes += [spike]
        return torch.stack(spikes,dim=-1), states


    def create_init_states(self):
        '''
        create initial states
        :return:
        '''
        device = self.snn_cell.v_0.device
        init_v = torch.zeros(self.neuron_number).to(device)
        init_reset_v = torch.zeros(self.neuron_number).to(device)
        init_psp_m = torch.zeros(self.input_size).to(device)
        init_psp_s = torch.zeros(self.input_size).to(device)

        init_states = (init_v, init_reset_v, init_psp_m, init_psp_s)

        return init_states

class synapse_cell(torch.nn.Module):
    '''
    implement spike response in synapse.
    i.e. every connection has its own time constants and states
    Although able to train it, this introduces very large computation overhead and memory consumption
    '''
    def __init__(self, input_size, output_size, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s):
        '''
        :param input_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param tau_s:
        :param train_tau_m:
        :param train_tau_s:
        '''
        super().__init__()
        self.input_size = input_size
        self.output_size= output_size
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau_m = torch.full((input_size,output_size), tau_m)
        self.tau_s = torch.full((input_size,output_size), tau_s)

        # calculate the norm factor to make max spike response to be 1
        eta = torch.tensor(tau_m / tau_s)
        self.v_0 = torch.nn.Parameter(torch.pow(eta, eta / (eta - 1)) / (eta - 1))
        self.v_0.requires_grad = False

        self.decay_m = torch.nn.Parameter(torch.exp(-1 / self.tau_m))
        self.decay_m.requires_grad = train_tau_m

        self.decay_s = torch.nn.Parameter(torch.exp(-1 / self.tau_s))
        self.decay_s.requires_grad = train_tau_s

        # self.transform = torch.nn.Linear(output_size, 1)

    def forward(self, current_spike, prev_states):
        """
        :param current_spike: [batch, idx]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
                prev_psp_m [input_size, output_size]
        :return:
        """

        prev_psp_m, prev_psp_s = prev_states

        # prev_psp_m/prev_psp_s shape is [batch, synapse_num, neuron_num]
        # current_spike shape is [batch, synapse_num], to broadcase,
        # need to unsqueeze it to [batch, synapse, 1]
        current_psp_m = prev_psp_m * self.decay_m + current_spike.unsqueeze(dim=-1)
        current_psp_s = prev_psp_s * self.decay_s + current_spike.unsqueeze(dim=-1)
        current_psp = current_psp_m - current_psp_s

        psp = current_psp * self.v_0

        new_states = (current_psp_m, current_psp_s)

        return psp, new_states

class synapse_layer(torch.nn.Module):
    '''
    wraper of synapse cell
    '''
    def __init__(self, input_size, output_size, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s):
        '''
        :param input_shape: tuple (width, hight) or (width, hight, depth) this shoule be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param tau_s:
        :param train_tau_m:
        :param train_tau_s:
        '''
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau_m = tau_m
        self.tau_s = tau_s

        self.synapse_cell = synapse_cell(input_size, output_size, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s)

    def forward(self, input_spikes, states):
        """
        :param input_spikes: [batch, dim0 ,dim1.., t]
        :param  states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):
            spike, states = self.synapse_cell(inputs[i], states)
            spikes += [spike]
        return torch.stack(spikes,dim=-1), states

    def create_init_states(self):

        device = self.synapse_cell.v_0.device
        init_psp_m = torch.zeros((self.batch_size, self.input_size, self.output_size)).to(device)
        init_psp_s = torch.zeros((self.batch_size, self.input_size, self.output_size)).to(device)

        init_states = (init_psp_m, init_psp_s)

        return init_states

class dual_exp_iir_cell(torch.nn.Module):
    '''
    implement spike response
    '''
    def __init__(self, input_shape, step_num, batch_size, tau_m, tau_s, train_coefficients):
        '''
        :param input_shape: tuple (width, hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param tau_s:
        :param train_coefficients:
        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau_m = torch.full(input_shape, tau_m)
        self.tau_s = torch.full(input_shape, tau_s)

        # calculate the norm factor to make max spike response to be 1
        eta = torch.tensor(tau_m / tau_s)
        self.v_0 = torch.nn.Parameter(torch.pow(eta, eta / (eta - 1)) / (eta - 1))
        self.v_0.requires_grad = False

        self.alpha_1 = torch.nn.Parameter(torch.exp(-1 / self.tau_m) + torch.exp(-1 / self.tau_s))
        self.alpha_1.requires_grad = train_coefficients

        self.alpha_2 = torch.nn.Parameter(-torch.exp( - (self.tau_m + self.tau_s)/(self.tau_m*self.tau_s) ))
        self.alpha_2.requires_grad = train_coefficients

        self.beta_1 = torch.nn.Parameter( torch.exp(-1/self.tau_m) - torch.exp(-1/self.tau_s))
        self.beta_1.requires_grad = train_coefficients

    def forward(self, current_spike, prev_states):
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """
        prev_t_1, prev_t_2 = prev_states

        current_psp = self.alpha_1 * prev_t_1 + self.alpha_2 * prev_t_2 + self.beta_1 * current_spike

        psp = current_psp

        new_states = (psp, prev_t_1)

        return psp, new_states

class dual_exp_iir_layer(torch.nn.Module):
    def __init__(self, input_shape, step_num, batch_size, tau_m, tau_s, train_coefficients):
        '''
        :param input_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param tau_s:
        :param train_coefficients:
        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau_m = tau_m
        self.tau_s = tau_s

        self.dual_exp_iir_cell = dual_exp_iir_cell(input_shape, step_num, batch_size, tau_m, tau_s, train_coefficients)

    def forward(self, input_spikes, states):
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  states: tuple (prev_t_1, prev_t_2)
        :return:
        """

        # unbind along last dimension
        inputs = list(input_spikes.unbind(dim=-1))
        spikes = []
        length = len(inputs)

        for i in range(length):
            spike, states = self.dual_exp_iir_cell(inputs[i], states)
            spikes += [spike]
        return torch.stack(spikes,dim=-1), states

    def create_init_states(self):

        device = self.dual_exp_iir_cell.v_0.device
        prev_t_1 = torch.zeros(self.input_shape).to(device)
        prev_t_2 = torch.zeros(self.input_shape).to(device)

        init_states = (prev_t_1, prev_t_2)

        return init_states

class first_order_low_pass_cell(torch.nn.Module):
    '''
    implement first order low pass filter
    '''
    def __init__(self, input_shape, step_num, batch_size, tau, train_coefficients):
        '''
        :param input_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau:

        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau = torch.full(input_shape, tau)

        self.alpha_1 = torch.nn.Parameter(torch.exp(-1 / self.tau))
        self.alpha_1.requires_grad = train_coefficients

    def forward(self, current_spike, prev_states):
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """
        prev_t_1 = prev_states

        current_psp = self.alpha_1 * prev_t_1 + current_spike

        psp = current_psp
        new_states = psp

        return psp, new_states

class first_order_low_pass_layer(torch.nn.Module):
    def __init__(self, input_shape, step_num, batch_size, tau, train_coefficients):
        '''
        :param input_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size
        self.tau = tau

        self.first_order_low_pass_cell = first_order_low_pass_cell(input_shape, step_num, batch_size, tau,
                                                    train_coefficients)

    def forward(self, input_spikes, states):
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  states: prev_psp
        :return:
        """

        # unbind along last dimension
        inputs = list(input_spikes.unbind(dim=-1))
        spikes = []
        length = len(inputs)

        for i in range(length):
            spike, states = self.first_order_low_pass_cell(inputs[i], states)
            spikes += [spike]
        return torch.stack(spikes, dim=-1), states

    def create_init_states(self):
        device = self.first_order_low_pass_cell.alpha_1.device
        prev_t_1 = torch.zeros(self.input_shape).to(device)

        init_states = prev_t_1

        return init_states

class axon_cell(torch.nn.Module):
    '''
    implement spike response in axon
    '''
    def __init__(self, input_shape, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s):
        '''
        :param input_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param tau_s:
        :param train_tau_m:
        :param train_tau_s:
        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau_m = torch.full(input_shape, tau_m)
        self.tau_s = torch.full(input_shape, tau_s)

        # calculate the norm factor to make max spike response to be 1
        eta = torch.tensor(tau_m / tau_s)
        self.v_0 = torch.nn.Parameter(torch.pow(eta, eta / (eta - 1)) / (eta - 1))
        self.v_0.requires_grad = False

        self.decay_m = torch.nn.Parameter(torch.exp(-1 / self.tau_m))
        self.decay_m.requires_grad = train_tau_m

        self.decay_s = torch.nn.Parameter(torch.exp(-1 / self.tau_s))
        self.decay_s.requires_grad = train_tau_s

    def forward(self, current_spike, prev_states):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        prev_psp_m, prev_psp_s = prev_states

        current_psp_m = prev_psp_m * self.decay_m + current_spike
        current_psp_s = prev_psp_s * self.decay_s + current_spike
        current_psp = current_psp_m - current_psp_s

        psp = current_psp * self.v_0

        new_states = (current_psp_m, current_psp_s)

        return psp, new_states

class axon_layer(torch.nn.Module):
    def __init__(self, input_shape, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s):
        '''

        :param input_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param tau_s:
        :param train_tau_m:
        :param train_tau_s:
        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau_m = tau_m
        self.tau_s = tau_s

        self.axon_cell = axon_cell(input_shape, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s)

    def forward(self, input_spikes, states):
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):
            spike, states = self.axon_cell(inputs[i], states)
            spikes += [spike]
        return torch.stack(spikes,dim=-1), states

    def create_init_states(self):

        device = self.axon_cell.v_0.device
        init_psp_m = torch.zeros(self.input_shape).to(device)
        init_psp_s = torch.zeros(self.input_shape).to(device)

        init_states = (init_psp_m, init_psp_s)

        return init_states

    def named_parameters(self, prefix='', recurse=True):
        '''
        return nothing
        :return:
        '''
        parameter_list = []
        for elem in parameter_list:
            yield elem


class neuron_cell(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m,
                train_bias, membrane_filter, input_type='axon', reset_v=1.0):
        '''
        :param input_size: int
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param membrane_filter: True or False
        '''
        super().__init__()
        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        if self.input_type != 'axon':
            weight = torch.empty([input_size,neuron_number])
            torch.nn.init.xavier_uniform(weight)
            self.input_type = 'synapse'
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = torch.nn.Linear(input_size, neuron_number, bias=train_bias)
        self.tau_m = torch.full((neuron_number,), tau_m)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full((self.neuron_number,),self.reset_decay))
        self.reset_decay.requires_grad = False

        self.reset_v = torch.nn.Parameter(torch.full((self.neuron_number,), reset_v))
        self.reset_v.requires_grad = False

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full((self.neuron_number,),self.decay_v))
        self.decay_v.requires_grad = False

        self.membrane_filter = membrane_filter

    def forward(self, current_input, prev_states):
        """
        :param current_input: [batch, input_size]
        :param  prev_states: tuple (prev_v, prev_reset)
        :return:
        """
        prev_v, prev_reset = prev_states

        if self.input_type != 'axon':
            weighted_psp = self.weight * current_input
            weighted_psp = weighted_psp.sum(dim=1)
        else:
            weighted_psp = self.weight(current_input)

        if self.membrane_filter:
            current_v = prev_v * self.decay_v + weighted_psp - prev_reset
        else:
            current_v = weighted_psp - prev_reset

        threshold_function = threshold.apply
        spike = threshold_function(current_v, self.sigma)

        current_reset = prev_reset * self.reset_decay + spike * self.reset_v

        new_states = (current_v, current_reset)

        return spike, new_states

class neuron_layer(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m,
    train_bias, membrane_filter, input_type='axon', reset_v=1.0):
        '''
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()
        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        self.neuron_cell = neuron_cell(input_size, neuron_number, step_num, batch_size, tau_m,
                                        train_bias, membrane_filter, input_type, reset_v)

    def forward(self, input_spikes, states):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  prev_states: tuple (prev_v, prev_reset)
        :return:
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):
            spike, states = self.neuron_cell(inputs[i], states)
            spikes += [spike]
        return torch.stack(spikes,dim=-1), states

    def create_init_states(self):

        device = self.neuron_cell.reset_decay.device
        init_v = torch.zeros(self.neuron_number).to(device)
        init_reset_v = torch.zeros(self.neuron_number).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

    def named_parameters(self, prefix='', recurse=True):
        '''
        only return weight in neuron cell
        '''
        for name, param in self.neuron_cell.weight.named_parameters(recurse=recurse):
            yield name, param

class neuron_cell_dot_product(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m,
                train_bias, membrane_filter, input_type='axon'):
        '''
        The the neuron performs dot product with input, not vector-matrix multiplication
        :param input_size: int
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param membrane_filter: True or False
        '''
        super().__init__()
        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        if self.input_type != 'axon':
            weight = torch.empty([input_size, neuron_number])
            torch.nn.init.xavier_uniform(weight)
            self.input_type = 'synapse'
            self.weight = torch.nn.Parameter(weight)
        else:
            weight = torch.empty([1,neuron_number])
            torch.nn.init.xavier_uniform(weight)
            self.weight = torch.nn.Parameter(weight)

        self.tau_m = torch.full((input_size,), tau_m)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0 / tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full((self.neuron_number,), self.reset_decay))
        self.reset_decay.requires_grad = False

        self.reset_v = torch.nn.Parameter(torch.full((self.neuron_number,), 1.0))
        self.reset_v.requires_grad = False

        self.decay_v = torch.exp(torch.tensor(-1 / tau_m))
        self.decay_v = torch.nn.Parameter(torch.full((self.neuron_number,), self.decay_v))
        self.decay_v.requires_grad = False

        self.membrane_filter = membrane_filter

    def forward(self, current_input, prev_states):
        """
        :param current_input: [batch, input_size]
        :param  prev_states: tuple (prev_v, prev_reset)
        :return:
        """
        prev_v, prev_reset = prev_states

        if self.input_type != 'axon':
            weighted_psp = self.weight * current_input
            weighted_psp = weighted_psp.sum(dim=1)
        else:

            weighted_psp = self.weight * current_input

        if self.membrane_filter:
            current_v = prev_v * self.decay_v + weighted_psp - prev_reset
        else:
            current_v = weighted_psp - prev_reset

        threshold_function = threshold.apply
        spike = threshold_function(current_v, self.sigma)

        current_reset = prev_reset * self.reset_decay + spike * self.reset_v

        new_states = (current_v, current_reset)

        return spike, new_states

class neuron_layer_dot_product(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m,
                train_bias, membrane_filter, input_type='axon'):
        '''

        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()
        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        self.neuron_cell = neuron_cell_dot_product(input_size, neuron_number, step_num, batch_size, tau_m,
                            train_bias, membrane_filter, input_type)


    def forward(self, input_spikes, states):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  prev_states: tuple (prev_v, prev_reset)
        :return:
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):
            spike, states = self.neuron_cell(inputs[i], states)
            spikes += [spike]
        return torch.stack(spikes,dim=-1), states

    def create_init_states(self):

        device = self.neuron_cell.reset_decay.device
        init_v = torch.zeros(self.neuron_number).to(device)
        init_reset_v = torch.zeros(self.neuron_number).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

class conv2d_cell(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num, batch_size,
                 tau_m, train_bias, membrane_filter, input_type='axon'):
        '''
        :param input_size: int
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param membrane_filter: True or False
        '''
        super().__init__()
        self.step_num = step_num
        self.batch_size = batch_size
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        self.h_input = h_input
        self.w_input =  w_input

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=train_bias)

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input,w_input,padding,kernel_size,stride)

        #output shape will be (batch, out_channels, conv_out_h, conv_out_w)
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full(self.output_shape,self.reset_decay))
        self.reset_decay.requires_grad = False

        self.reset_v = torch.nn.Parameter(torch.full(self.output_shape, 1.0))
        self.reset_v.requires_grad = False

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full(self.output_shape,self.decay_v))
        self.decay_v.requires_grad = False

        self.membrane_filter = membrane_filter

    def forward(self, current_input, prev_states):
        """
        :param current_input: [batch, input_size]
        :param  prev_states: tuple (prev_v, prev_reset)
        :return:
        """
        prev_v, prev_reset = prev_states

        conv2d_out = self.conv(current_input)

        if self.membrane_filter:
            current_v = prev_v * self.decay_v + conv2d_out - prev_reset
        else:
            current_v = conv2d_out - prev_reset

        threshold_function = threshold.apply
        spike = threshold_function(current_v, self.sigma)

        current_reset = prev_reset * self.reset_decay + spike * self.reset_v

        new_states = (current_v, current_reset)

        return spike, new_states

class conv2d_layer(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num, batch_size,
                 tau_m, train_bias, membrane_filter, input_type='axon'):
        '''
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()
        # self.input_size = input_size
        # self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input, w_input, padding, kernel_size, stride)
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        self.conv2d_cell = conv2d_cell(h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num, batch_size,
                 tau_m, train_bias, membrane_filter, input_type)

    def forward(self, input_spikes, states):
        """
        :param input_spikes: [batch, dim0 ,dim1..,t]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):
            spike, states = self.conv2d_cell(inputs[i], states)
            spikes += [spike]
        return torch.stack(spikes,dim=-1), states

    def create_init_states(self):

        device = self.conv2d_cell.reset_decay.device

        init_v = torch.zeros(self.output_shape).to(device)
        init_reset_v = torch.zeros(self.output_shape).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

    def named_parameters(self, prefix='', recurse=True):
        '''
        only return weight in neuron cell
        '''
        for name, param in self.conv2d_cell.conv.named_parameters(recurse=recurse):
            yield name, param

class maxpooling2d_layer(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, kernel_size, stride, padding, dilation, step_num, batch_size):
        '''
        2d max pooling, input should be the output of axon, it pools the axon's psp
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        '''
        super().__init__()
        # self.input_size = input_size
        # self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        # self.input_type = input_type

        self.maxpooling2d = torch.nn.MaxPool2d(kernel_size,stride,padding,dilation)

        self.output_shape = calculate_maxpooling2d_outsize(h_input, w_input, padding, kernel_size, stride)


    def forward(self, input_psp):
        """
        :param input_spikes: [batch, dim0 ,dim1..,t]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        # unbind along last dimension
        inputs = input_psp.unbind(dim=-1)
        pooled_psp = []
        for i in range(len(inputs)):
            psp = self.maxpooling2d(inputs[i])
            pooled_psp += [psp]
        return torch.stack(pooled_psp,dim=-1)


def calculate_conv2d_outsize(h_input, w_input, padding, kernel_size, stride, dilation=1):
    '''
    calculate the output size of conv2d
    :param h_input:
    :param w_input:
    :param padding:
    :param dilation:
    :param kernel_size:
    :return:
    '''
    h_output = (h_input + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
    w_output = (w_input + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1

    return h_output, w_output

def calculate_maxpooling2d_outsize(h_input, w_input, padding, kernel_size, stride, dilation=1):
    '''
    calculate the output size of conv2d
    :param h_input:
    :param w_input:
    :param padding:
    :param dilation:
    :param kernel_size:
    :return:
    '''
    h_output = (h_input + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
    w_output = (w_input + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1

    return h_output, w_output


class SNN_Monitor():
    """
    Record spikes and states
    reference: https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
    """
    def __init__(self, module, max_iteration = 1):

        self.step_num = module.step_num
        self.max_iteration = max_iteration

        self.variable_dict = {}
        self.record = {}

        self.counter  = 0
        self.max_len = max_iteration * self.step_num

        if isinstance(module, dual_exp_iir_layer):
            self.psp_list = []
            self.hook = module.dual_exp_iir_cell.register_forward_hook(self.get_output_dual_exp_iir)

            self.variable_dict['psp'] = self.psp_list

        elif isinstance(module, first_order_low_pass_layer):
            self.psp_list = []
            self.hook = module.first_order_low_pass_cell.register_forward_hook(self.get_output_first_order_low_pass)

            self.variable_dict['psp'] = self.psp_list

        elif isinstance(module, neuron_layer):
            self.hook = module.neuron_cell.register_forward_hook(self.get_output_neuron_layer)
            self.v_list = []
            self.reset_v_list = []
            self.spike_list = []

            self.variable_dict['v'] = self.v_list
            self.variable_dict['reset_v'] = self.reset_v_list
            self.variable_dict['spike'] = self.spike_list

        elif isinstance(module, axon_layer):
            self.hook =module.axon_cell.register_forward_hook(self.get_output_axon_layer)
            self.psp_list = []

            self.variable_dict['psp'] = self.psp_list

    def get_output_dual_exp_iir(self, module, input, output):
        '''

        :param module:
        :param input: a tuple [spike, new state[state(t-1), state(t-2)]]
        :param output: a tuple [psp, new state[psp, state(t-1)]]
        :return:
        '''

        self.counter += 1
        if self.counter > self.max_len:
            return

        self.psp_list.append(output[0])

        if self.counter == self.max_len:
            self.reshape()

    def get_output_first_order_low_pass(self, module, input, output):
        '''

        :param module:
        :param input:
        :param output:
        :return:
        '''

        self.counter += 1
        if self.counter > self.max_len:
            return

        self.psp_list.append(output[0])

        if self.counter == self.max_len:
            self.reshape()

    def get_output_neuron_layer(self, module, input, output):
        '''

        :param module:
        :param input:
        :param output: [spike, [v, reset_v]]
        :return:
        '''

        self.counter += 1
        if self.counter > self.max_len:
            return

        self.spike_list.append(output[0])
        self.v_list.append(output[1][0])
        self.reset_v_list.append(output[1][1])

        if self.counter == self.max_len:
            self.reshape()

    def get_output_axon_layer(self, module, input, output):
        '''

        :param module:
        :param input:
        :param output:
        :return:
        '''

        self.counter += 1
        if self.counter > self.max_len:
            return

        self.psp_list.append(output[0])

        if self.counter == self.max_len:
            self.reshape()

    def reshape(self):

        for key in self.variable_dict:
            temp_list = []
            for element in self.variable_dict[key]:
                temp_list.append(element.detach().cpu().numpy())

            # shape packed [total steps, batch, neuron/synapse]
            packed = np.array(temp_list)

            #shape packed [iterations,step num, batch, neuron/synapse]
            packed = np.reshape(packed, (self.max_iteration, self.step_num, *packed.shape[1:]))

            self.record[key] = packed
