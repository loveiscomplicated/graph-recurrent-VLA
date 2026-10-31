"""Preserved E0/GRU policy; geometry targets are loss-only arguments."""
import torch
from graph.plan_a_bc.model import CachedGraphPolicy
from graph.plan_a_bc.common import tensor_identity, digest
from graph.gt_pretrain10f1.model import SpatialHead


class Policy(CachedGraphPolicy):
    def __init__(self, schema, f1, condition, seed=42):
        # Initialize every control parameter before constructing the auxiliary head.
        torch.manual_seed(seed)
        super().__init__(schema, 'E0', d=64, L=2, gru_hidden=144, gru_layers=1, enc_hidden=128)
        self.condition = condition
        self.graph_encoder.load_state_dict({k.removeprefix('graph_encoder.'): v
            for k, v in f1['model'].items() if k.startswith('graph_encoder.')}, strict=True)
        self.f1_stats = f1['normalization']
        if condition == 'C':
            self.state_head = SpatialHead(398, self.f1_stats['target_mean'])
            self.state_head.load_state_dict({k.removeprefix('state_head.'): v
                for k,v in f1['model'].items() if k.startswith('state_head.')}, strict=True)
        elif condition not in ('A', 'B'):
            raise ValueError(condition)
        if condition == 'A':
            self.graph_encoder.requires_grad_(False)
        self.last_readout = self.last_presence = None

    def fuse(self, graphs, robot_state, normalizer, trace=None):
        if normalizer.schema != self.robot_schema:
            raise ValueError('robot schema mismatch')
        z, presence = self.graph_encoder(graphs, trace)
        self.last_readout, self.last_presence = z, presence
        return torch.cat((z, presence, normalizer.apply(robot_state)), -1)

    def geometry_prediction(self, z, presence, robot):
        if self.condition != 'C':
            raise ValueError('auxiliary training head only exists in C')
        mean = robot.new_tensor(self.f1_stats['robot_mean'])
        std = robot.new_tensor(self.f1_stats['robot_std'])
        return self.state_head(torch.cat((z, presence, (robot-mean)/std), -1))

    def initial_hashes(self):
        return dict(gnn=tensor_identity(self.graph_encoder.state_dict()),
            readout=digest(dict(slots=['TCP','Left','Right','Peg','Box'],width=64,reduction='parameter-free mean')), 
            projection=tensor_identity(self.policy.encoder.state_dict()),
            gru=tensor_identity(self.policy.rnn.state_dict()),
            action_head=tensor_identity(self.policy.head.state_dict()),
            state_head=tensor_identity(self.state_head.state_dict()) if self.condition=='C' else None)
