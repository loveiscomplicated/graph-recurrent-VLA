"""Sensor-only inference and the same Task10G recurrent control topology."""
import numpy as np
import torch
from torch import nn
from recurrent.models import RNNPolicy
from graph.gt_pretrain10f1.model import restore_checkpoint
from graph.gt_pretrain10f1.contract import split_fields, measured_robot, assemble_gt38
from graph.plan_a_bc.common import tensor_identity


class Policy(nn.Module):
    def __init__(self, checkpoint, schema, statistics, seed=42, device='cuda'):
        super().__init__()
        self.frozen, ck = restore_checkpoint(checkpoint, device, freeze=True)
        if ck['robot_schema'] != schema:
            raise ValueError('F1 Robot76 schema differs')
        self.robot_schema = schema
        # Explicit fresh RNG initialization; no controller checkpoint is read.
        torch.manual_seed(seed)
        self.initial_rng = torch.get_rng_state().clone()
        self.policy = RNNPolicy(38, 7, cell='gru', hidden=144, layers=1, enc_hidden=128).to(device)
        self.register_buffer('state_mean', torch.tensor(statistics['mean'], dtype=torch.float32, device=device))
        self.register_buffer('state_std', torch.tensor(statistics['std'], dtype=torch.float32, device=device))
        self.condition = 'S'
        self.last_state38 = None

    @property
    def graph_encoder(self):
        return self.frozen.graph_encoder

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    @torch.no_grad()
    def extract_state(self, graph, robot):
        # Reuse the F1 prediction and exact float32 assembly arithmetic.
        obj, _, _ = self.frozen(graph, robot)
        state = assemble_gt38(split_fields(obj.detach().cpu().numpy()),
                              measured_robot(robot.detach().cpu().numpy(), self.robot_schema))
        self.last_state38 = state.copy()
        return torch.as_tensor(state, dtype=torch.float32, device=robot.device)

    def normalize(self, state):
        if state.shape[-1] != 38 or state.dtype != torch.float32 or not torch.isfinite(state).all():
            raise ValueError('finite float32 State38 only')
        return (state-self.state_mean)/self.state_std

    def from_states(self, states, hidden=None, occupied=None):
        x = self.normalize(states)
        if occupied is None or bool(occupied.all()):
            return self.policy(x, hidden)
        # Same inactive-stream behavior as IntegrationPolicy.forward.
        b, t, _ = x.shape
        if hidden is None:
            hidden = x.new_zeros((1, b, 144))
        outputs = []
        for i in range(t):
            out, nh = self.policy(x[:, i:i+1], hidden)
            active = occupied[:, i]
            hidden = torch.where(active[None, :, None], nh, hidden)
            outputs.append(out[:, 0]*active[:, None])
        return torch.stack(outputs, 1), hidden

    def forward(self, graphs, robot, normalizer=None, hidden=None):
        state = self.extract_state(graphs, robot)
        fm = graphs.frame_map
        b, t = int(fm[:, 0].max())+1, int(fm[:, 1].max())+1
        seq = state.new_zeros((b, t, 38))
        occupied = torch.zeros((b, t), dtype=torch.bool, device=state.device)
        seq[fm[:, 0], fm[:, 1]] = state
        occupied[fm[:, 0], fm[:, 1]] = True
        lengths = occupied.sum(1)
        if not torch.equal(occupied, torch.arange(t, device=state.device)[None] < lengths[:, None]):
            raise ValueError('noncontiguous sequence')
        return self.from_states(seq, hidden, occupied)

    def step(self, graphs, robot, normalizer=None, hidden=None):
        if not bool((graphs.frame_map[:, 1] == 0).all()):
            raise ValueError('one observation per stream')
        out, hidden = self(graphs, robot, normalizer, hidden)
        return out[:, 0], hidden

    def frozen_hash(self):
        return tensor_identity(self.frozen.state_dict())

    def control_hash(self):
        return tensor_identity(self.policy.state_dict())


def fit_normalizer(train):
    """Train predicted frames only; floor in native units; no clipping or GT fit."""
    if train.ndim != 2 or train.shape[1] != 38 or not np.isfinite(train).all():
        raise ValueError('finite predicted train38 required')
    # Position, shape, finger and radius dimensions: 1mm floor.
    # Quaternions: 0.01 unitless floor, retaining their native wxyz values.
    floor = np.full(38, .001, np.float64)
    floor[[3,4,5,6,18,19,20,21,28,29,30,31]] = .01
    mean = train.astype(np.float64).mean(0)
    raw = train.astype(np.float64).std(0)
    std = np.maximum(raw, floor)
    # Explicit clean constants normalize to zero with safe unit scale.
    mean[36:] = [1., 0.]
    std[36:] = 1.
    return dict(mean=mean.astype('f').tolist(), std=std.astype('f').tolist(),
        raw_std=raw.tolist(), floor=floor.tolist(), fitted_split='train',
        fitted_frames=len(train), input='F1-predicted State38 + measured robot only',
        formula='(state38 - float32(mean)) / float32(std), before Projection38->128; no clipping',
        variance='float64 population moments ddof=0, stored float32',
        quaternion='native normalized canonical wxyz; no additional transform; std floor0.01',
        positive_shape='native positive F1 prediction; no log, GT size, clamp or correction',
        constant_freshness_age='mean=[1,0], std=[1,1] -> [0,0]',
        smallest_position_scale_m=.001, final_used=False)
