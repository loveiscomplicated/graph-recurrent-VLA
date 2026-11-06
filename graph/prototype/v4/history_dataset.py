"""Episode-local s[t-K:t+1], a[t-K:t]; target a[t] never enters inputs."""
import torch
from .devices import move_inputs


class HistoryDataset:
    def __init__(self, data, length=4):
        if length < 0:
            raise ValueError('history length must be nonnegative')
        self.data, self.length = data, length
        n = len(data['actions'])
        self.states = torch.empty(n, length+1, dtype=torch.long)
        self.past = torch.zeros(n, length, dtype=torch.long)
        self.valid = torch.zeros(n, length, dtype=torch.bool)
        for eid in torch.unique(data['episode_id']):
            rows = torch.where(data['episode_id'] == eid)[0]
            rows = rows[torch.argsort(data['timestep'][rows])]
            if not torch.equal(data['timestep'][rows], torch.arange(len(rows))):
                raise ValueError('episode timesteps must be contiguous from zero')
            offsets = torch.arange(len(rows))[:, None] + torch.arange(-length, 1)[None]
            self.states[rows] = rows[offsets.clamp_min(0)]
            self.past[rows] = rows[offsets[:, :-1].clamp_min(0)]
            self.valid[rows] = offsets[:, :-1] >= 0

    def window(self, ids, state_history=True):
        states = self.states[ids] if state_history else ids[:, None]
        valid = self.valid[ids]
        actions = self.data['actions'][self.past[ids]].clone()
        actions[~valid] = 0
        return states, actions, valid.float()

    def forward(self, model, ids, geometry, qvel, norm, sn, builder):
        states, actions, valid = self.window(ids, model.use_state_history)
        # One shared encoder; repeated padding rows are evaluated once. Indexing
        # retains autograd to every real timestep during training.
        unique, inverse = torch.unique(states.flatten(), sorted=True, return_inverse=True)
        g = geometry.select(unique)
        device = next(model.parameters()).device
        h = model.encode(move_inputs(norm.apply(model.name, g, builder), device), sn.apply(g, qvel[unique], 'goal').to(device))
        h = h[inverse.to(device)].reshape(len(ids), states.shape[1], -1)
        return model.from_history(h, actions.to(device), valid.to(device))
