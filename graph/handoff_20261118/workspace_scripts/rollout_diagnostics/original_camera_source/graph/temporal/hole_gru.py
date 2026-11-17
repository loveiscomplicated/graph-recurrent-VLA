"""Causal GRU hole-position estimator, independent of the Action7 controller."""

import math

import torch
from torch import nn

from graph.gated_mpnn import assemble_state38
from graph.temporal_gate import correct_object21
from graph.temporal.observations import OBSERVATION_NAMES, temporal_inputs


class TemporalHoleGRU(nn.Module):
    """Map current observations and episode-local hidden state to world XYZ.

    A direct hidden-to-position head can recover even when the current GNN is
    wrong. There is no residual shortcut, forced first prediction, observation
    threshold, teacher forcing or bidirectional recurrence.
    """

    def __init__(self, hidden=64, layers=1, position_scale_m=0.1):
        super().__init__()
        if type(hidden) is not int or hidden < 1:
            raise ValueError('gru.hidden must be a positive integer (initial candidates: 64, 128)')
        if layers != 1 or type(layers) is not int:
            raise ValueError('gru.layers must be 1 for this feasibility experiment')
        if (type(position_scale_m) not in (int, float)
                or not math.isfinite(position_scale_m) or position_scale_m <= 0):
            raise ValueError('gru.position_scale_m must be finite and positive')
        self.hidden = hidden
        self.position_scale_m = position_scale_m
        self.gru = nn.GRU(len(OBSERVATION_NAMES), hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, observations, hidden=None):
        """Return world positions[B,T,3], hidden[1,B,H]; None starts a new episode.

        Every row of a batch must be one complete sequence, without padding.
        Training uses batch size one so variable episode lengths need no mask.
        """
        if (observations.ndim != 3 or observations.shape[-1] != len(OBSERVATION_NAMES)
                or observations.shape[0] < 1 or observations.shape[1] < 1):
            raise ValueError('observations must have nonempty shape [B,T,23]')
        if not observations.is_floating_point() or not torch.isfinite(observations).all():
            raise ValueError('observations must be finite floating point values')
        if hidden is not None:
            if hidden.shape != (1, observations.shape[0], self.hidden):
                raise ValueError(f'hidden must have shape [1,B,{self.hidden}] or be None')
            if not torch.isfinite(hidden).all():
                raise ValueError('hidden must be finite')
        scaled = torch.cat((observations[..., :6] / self.position_scale_m,
                            observations[..., 6:]), dim=-1)
        states, hidden = self.gru(scaled, hidden)
        return self.head(states) * self.position_scale_m, hidden

    def step(self, observation, hidden=None):
        """Advance observations[B,23]; pass hidden=None at every episode reset."""
        if observation.ndim != 2:
            raise ValueError('step observation must have shape [B,23]')
        world, hidden = self.forward(observation.unsqueeze(1), hidden)
        return world[:, 0], hidden


class StreamingHoleEstimator:
    """Own one episode's hidden state; reset explicitly at simulator episode reset.

    Example: estimator.reset(); result = estimator.step(object21, tcp, quality).
    All arguments are batched device tensors; quality is observation_features().
    Robot15 and freshness2 are supplied by the caller when State38 is required.
    This wrapper never stores GT or creates robot/freshness values from labels.
    """

    def __init__(self, model):
        self.model = model.eval()
        self.reset()

    def reset(self):
        """Release hidden state so the next frame starts from zero memory."""
        self.hidden = None

    @torch.inference_mode()
    def step(self, object21, tcp_world, quality, robot15=None, freshness2=None):
        """Return hole-only Object21 and a separate geometry-consistent assembly."""
        if (robot15 is None) != (freshness2 is None):
            raise ValueError('Supply robot15 and freshness2 together to assemble State38')
        inputs = temporal_inputs(object21, tcp_world, quality)
        world, self.hidden = self.model.step(inputs, self.hidden)
        relative = world - tcp_world
        result = {'hole_world': world,
                  'object21': correct_object21(object21, relative, preserve_endpoint=False),
                  'consistent_object21': correct_object21(object21, relative)}
        if robot15 is not None:
            result['state38'] = assemble_state38(robot15, result['consistent_object21'], freshness2)
        return result
