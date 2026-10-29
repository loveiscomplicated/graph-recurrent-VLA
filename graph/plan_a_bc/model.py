"""Existing Task08 GRU/head, with09-D Graph21 E0 or E1 at the same readout."""
import torch
from torch import nn
from graph.integration_baseline.model import IntegrationPolicy
from graph.prototype.models import mlp
from graph.integration_baseline.robot import RobotNormalizer
from graph.prototype.v2.state import StateNormalizer
from .common import read
from pathlib import Path


def normalizer(root):
    s=read(Path(root)/'normalizer.json')
    return RobotNormalizer(s['schema'],StateNormalizer(torch.tensor(s['mean']),torch.tensor(s['std'])),s['fitted_episode_ids'])


class CachedGraphPolicy(IntegrationPolicy):
    def __init__(self,schema,variant,**config):
        if variant not in ('E0','E1'):raise ValueError('E0/E1 only')
        super().__init__(schema,**config)
        self.variant=variant
        self.graph_encoder.node_encoder=mlp([21+4,config['d'],config['d']],final_activation=True)
        if variant=='E1':self.graph_encoder.layers=nn.ModuleList()


def gradient_report(model):
    result={}
    for name,prefix in [('encoder','graph_encoder.'),('gru','policy.rnn.'),('head','policy.head.')]:
        values=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if not values or not all(torch.isfinite(v).all() for v in values):raise FloatingPointError('missing/nonfinite '+name+' gradients')
        norm=float(torch.linalg.vector_norm(torch.stack([v.norm() for v in values])))
        if norm==0:raise ValueError('zero '+name+' gradients')
        result[name]=norm
    if model.variant=='E0':
        values=[p.grad for p in model.graph_encoder.layers.parameters()]
        if not values or any(v is None for v in values):raise ValueError('message passing gradient missing')
    return result
