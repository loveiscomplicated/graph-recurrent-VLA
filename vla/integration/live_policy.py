"""Existing Task08 GRU/head, with09-D Graph21 E0 or E1 at the same readout."""
import torch
from vla.integration.policy import IntegrationPolicy
from gnn.edge_mlp import configure_live_encoder
from vla.data.robot import RobotNormalizer
from vla.data.normalization import StateStatistics as StateNormalizer
from vla.data.files import read
from pathlib import Path


def normalizer(root):
    s=read(Path(root)/'normalizer.json')
    return RobotNormalizer(s['schema'],StateNormalizer(torch.tensor(s['mean']),torch.tensor(s['std'])),s['fitted_episode_ids'])


class CachedGraphPolicy(IntegrationPolicy):
    def __init__(self,schema,variant,**config):
        if variant not in ('E0','E1'):raise ValueError('E0/E1 only')
        self.baseline_encoder = config.get('graph_encoder') is None
        super().__init__(schema,**config)
        self.variant=variant
        if self.baseline_encoder:
            configure_live_encoder(self.graph_encoder,d=config['d'],variant=variant)


def gradient_report(model):
    result={}
    groups={'encoder':model.graph_encoder}
    from controller import RNNPolicy
    if isinstance(model.policy,RNNPolicy):
        groups.update(gru=model.policy.rnn,head=model.policy.head)
    else:
        groups['controller']=model.policy
    for name,module in groups.items():
        trainable=[p for p in module.parameters() if p.requires_grad]
        if not trainable:
            result[name]=0.;continue
        values=[p.grad for p in trainable if p.grad is not None]
        if not values or not all(torch.isfinite(v).all() for v in values):raise FloatingPointError('missing/nonfinite '+name+' gradients')
        norm=float(torch.linalg.vector_norm(torch.stack([v.norm() for v in values])))
        if norm==0:raise ValueError('zero '+name+' gradients')
        result[name]=norm
    if model.baseline_encoder and model.variant=='E0' and any(p.requires_grad for p in model.graph_encoder.layers.parameters()):
        values=[p.grad for p in model.graph_encoder.layers.parameters()]
        if not values or any(v is None for v in values):raise ValueError('message passing gradient missing')
    return result
