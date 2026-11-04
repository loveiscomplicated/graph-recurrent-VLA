"""Current graph/controller integration, extracted without changing baseline weights."""
from time import perf_counter
import torch
from torch import nn
from gnn.edge_mlp import GraphEncoder
from controller.recurrent import RNNPolicy
from vla.state import reset_state, select_state


class IntegrationPolicy(nn.Module):
    def __init__(self, robot_schema, *, d=64, L=2, gru_hidden=144, gru_layers=1, enc_hidden=128,
                 graph_encoder=None, controller=None, graph_feature_dim=None,
                 presence_dim=2, state_batch_dim=None):
        super().__init__()
        self.robot_schema = robot_schema
        self.graph_encoder = GraphEncoder(d, L) if graph_encoder is None else graph_encoder
        self.graph_feature_dim = (getattr(self.graph_encoder, 'output_dim', 5*d)
                                  if graph_feature_dim is None else graph_feature_dim)
        if type(self.graph_feature_dim) is not int or self.graph_feature_dim < 1:
            raise ValueError('positive graph_feature_dim required')
        if type(presence_dim) is not int or presence_dim < 0:
            raise ValueError('nonnegative presence_dim required')
        self.presence_dim = presence_dim
        self.fused_dim = self.graph_feature_dim + presence_dim + robot_schema['dimension']
        self.policy = (RNNPolicy(self.fused_dim, 7, cell='gru', hidden=gru_hidden,
                                layers=gru_layers, enc_hidden=enc_hidden)
                       if controller is None else controller)
        if getattr(self.policy, 'input_dim', self.fused_dim) != self.fused_dim:
            raise ValueError('controller input dimension does not match graph/state fusion')
        if getattr(self.policy, 'action_dim', 7) != 7:
            raise ValueError('current live action contract requires seven values')
        self.state_batch_dim = (getattr(self.policy, 'state_batch_dim', 1)
                                if state_batch_dim is None else state_batch_dim)

    def fuse(self, graphs, robot_state, normalizer, trace=None):
        if normalizer.schema != self.robot_schema:
            raise ValueError('normalizer/model schema mismatch')
        if robot_state.shape != (graphs.count, self.robot_schema['dimension']):
            raise ValueError('one full robot state per mapped graph required')
        z, presence = self.graph_encoder(graphs, trace)
        if z.shape != (graphs.count, self.graph_feature_dim):
            raise ValueError('graph encoder feature dimension does not match integration')
        if presence.shape != (graphs.count, self.presence_dim):
            raise ValueError('graph encoder presence dimension does not match integration')
        tick = perf_counter()
        fused = torch.cat((z, presence, normalizer.apply(robot_state)), -1)
        if trace is not None:
            trace.update(fusion_ms=(perf_counter()-tick)*1000, fused_shape=list(fused.shape))
        return fused

    def forward(self, graphs, robot_state, normalizer, hidden=None, *, lengths=None, reset_before=None, trace=None):
        fused = self.fuse(graphs, robot_state, normalizer, trace)
        b = int(graphs.frame_map[:, 0].max())+1
        t = int(graphs.frame_map[:, 1].max())+1
        sequence = fused.new_zeros((b, t, self.fused_dim))
        sequence[graphs.frame_map[:, 0], graphs.frame_map[:, 1]] = fused
        occupied = torch.zeros((b, t), dtype=torch.bool, device=fused.device)
        occupied[graphs.frame_map[:, 0], graphs.frame_map[:, 1]] = True
        if lengths is None:
            lengths = occupied.sum(1)
        lengths = torch.as_tensor(lengths, device=fused.device)
        if lengths.shape != (b,) or not torch.equal(occupied, torch.arange(t, device=fused.device)[None] < lengths[:, None]):
            raise ValueError('sequence mapping must be contiguous, without holes or trailing fake frames')
        if reset_before is None and occupied.all():
            return self.policy(sequence, hidden)
        if reset_before is None:
            reset_before = torch.zeros((b, t), dtype=torch.bool, device=fused.device)
        if reset_before.shape != (b, t):
            raise ValueError('reset mask must be [B,T]')
        outputs = []
        for i in range(t):
            active = occupied[:, i]
            hidden = reset_state(hidden, reset_before[:, i] & active,
                                 batch_dim=self.state_batch_dim)
            out, next_hidden = self.policy(sequence[:, i:i+1], hidden)
            hidden = select_state(hidden, next_hidden, active, batch_dim=self.state_batch_dim)
            outputs.append(out[:, 0] * active[:, None])
        return torch.stack(outputs, 1), hidden

    def step(self, graphs, robot_state, normalizer, hidden=None, reset=None):
        b = graphs.count
        if not torch.equal(graphs.frame_map, torch.stack((torch.arange(b, device=graphs.frame_map.device),
                                                         torch.zeros(b, dtype=torch.long, device=graphs.frame_map.device)), -1)):
            raise ValueError('step requires one graph per stream with timestep slot zero')
        if hidden is not None and reset is not None:
            hidden = reset_state(hidden, reset, batch_dim=self.state_batch_dim)
        action, hidden = self.forward(graphs, robot_state, normalizer, hidden)
        return action[:, 0], hidden


class EpisodeStream:
    """Single stream lifecycle; episode changes/flags/reset clear hidden state."""
    def __init__(self, model, normalizer):
        self.model, self.normalizer = model, normalizer
        self.hidden = None
        self.identity = None
        self.timestep = None

    def reset(self):
        self.hidden = self.identity = self.timestep = None

    def step(self, graphs, robot_state, episode_identity, timestep, *, reset=False,
             previous_terminated=False, previous_truncated=False):
        if reset or previous_terminated or previous_truncated or self.identity != episode_identity:
            self.reset()
        if self.timestep is not None and timestep != self.timestep+1:
            raise ValueError('noncontiguous stream observation')
        action, self.hidden = self.model.step(graphs, robot_state, self.normalizer, self.hidden)
        self.identity, self.timestep = episode_identity, timestep
        return action
