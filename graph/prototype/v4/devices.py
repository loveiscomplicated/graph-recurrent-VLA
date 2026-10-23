"""CPU graph construction, explicit network device, portable checkpoints."""
from dataclasses import replace
import torch


def move_inputs(inputs, device):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    return replace(inputs, **{k:v.to(device) for k,v in vars(inputs).items() if isinstance(v,torch.Tensor)})


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def synchronize(device):
    if str(device)=='mps':torch.mps.synchronize()
