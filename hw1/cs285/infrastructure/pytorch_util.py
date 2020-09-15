from typing import Union

import numpy as np
import torch
from torch import nn
from cs285.policies.MLP_policy import MLPPolicy, MLPPolicySL

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, num_layers, layer_size, in_activation, out_activation):
        super(MLP, self).__init__()
        self.start_layer = torch.nn.Linear(in_size, layer_size)
        self.layers = [self.start_layer]
        self.in_activation = in_activation
        self.out_activation = out_activation
        for i in range(num_layers):
            self.layers.append(torch.nn.Linear(layer_size, layer_size))

        self.out_layer = torch.nn.Linear(layer_size, out_size)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = from_numpy(x)

        for layer in self.layers:
            x = self.in_activation(layer(x))

        return self.out_activation(self.out_layer(x))


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.

    return MLP(input_size, output_size, n_layers, size, activation, output_activation)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
