"""Neural building blocks preserved from the live baseline."""
from torch import nn


def mlp(dims, final_activation=False):
    layers = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
