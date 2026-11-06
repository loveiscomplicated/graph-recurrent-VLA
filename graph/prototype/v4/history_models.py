"""Fixed window feed-forward policy with the unchanged v2 encoder."""
import torch
from ..models import mlp
from ..v2.models import CommonPolicy

MODES = {'current': (False, False), 'action': (False, True),
         'state': (True, False), 'both': (True, True)}


class HistoryPolicy(CommonPolicy):
    def __init__(self, name, cfg, history='both', length=4):
        super().__init__(name, cfg, 'goal', 'regression')
        if history not in MODES or length < 0:
            raise ValueError('invalid history configuration')
        self.name, self.history, self.length = name, history, length
        self.use_state_history, self.use_action_history = MODES[history]
        self.state_count = length+1 if self.use_state_history else 1
        self.mask_dim = length if history != 'current' else 0
        self.input_dim = 159*self.state_count + 7*length*self.use_action_history + self.mask_dim
        if history != 'current':
            self.action_head = mlp([self.input_dim, cfg.get('head_width', 128), 7])

    def encode(self, inputs, shared):
        z = self.encoder(inputs)
        if z.shape[1] != 128 or shared.shape != (len(z), 31):
            raise ValueError('v2 representation shape changed')
        return torch.cat((z, shared), -1)

    def from_history(self, states, actions, valid):
        b = len(states)
        if states.shape != (b, self.state_count, 159) or actions.shape != (b, self.length, 7) or valid.shape != (b, self.length):
            raise ValueError('history shapes do not match configuration')
        pieces = [states.flatten(1)]
        if self.use_action_history:
            pieces.append(actions.flatten(1))
        if self.mask_dim:
            pieces.append(valid)
        return self.action_head(torch.cat(pieces, -1))
