"""Shared GRU temporal Peg/Hole pose estimator; the Action7 controller is separate."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from graph.gated_mpnn import assemble_state38
from graph.pose_geometry import (geometric_endpoint, hole_frame_vector, peg_axis, quaternion_product,
                                  relative_rotation, rotvec_quaternion, unit_quaternion)
from graph.temporal.pose_observations import BASE_NAMES, ROBOT_NAMES, pose_input_step

MODES = {'hole_only': ('hole',), 'peg_only': ('peg',), 'joint_peg_hole': ('peg', 'hole')}


class TemporalPoseGRU(nn.Module):
    """Project observations, share temporal context and decode separate pose heads.

    Input Object21 positions are in world coordinates. Position heads support
    direct world regression or residual correction; rotation heads predict a
    world-axis rotation vector, left-composed with the current GNN quaternion.
    Normalization buffers are fitted on training observations only and checkpointed.
    """

    def __init__(self, mode='joint_peg_hole', with_robot=False, hidden=128,
                 projection=64, head_hidden=64, layers=1, position_mode='direct', position_scale_m=.1):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'mode must be one of {tuple(MODES)}')
        for name, value in (('hidden', hidden), ('projection', projection), ('head_hidden', head_hidden)):
            if type(value) is not int or value < 1:
                raise ValueError(f'gru.{name} must be a positive integer')
        if type(layers) is not int or layers != 1:
            raise ValueError('gru.layers must be 1')
        if position_mode not in ('direct', 'residual'):
            raise ValueError('gru.position_mode must be direct or residual')
        if not 0 < position_scale_m < float('inf'):
            raise ValueError('gru.position_scale_m must be finite and positive')
        self.mode, self.with_robot = mode, with_robot
        self.hidden, self.position_mode = hidden, position_mode
        self.position_scale_m = position_scale_m
        width = len(BASE_NAMES) + (len(ROBOT_NAMES) if with_robot else 0)
        self.register_buffer('input_mean', torch.zeros(width))
        self.register_buffer('input_std', torch.ones(width))
        self.projection = nn.Sequential(nn.Linear(width, projection), nn.SiLU())
        self.gru = nn.GRU(projection, hidden, num_layers=layers, batch_first=True)
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(hidden, head_hidden), nn.SiLU(),
                                                       nn.Linear(head_hidden, 6)) for name in MODES[mode]})

    def set_normalization(self, mean, std):
        """Copy train-only statistics; no per-episode or test-time adaptation."""
        if (mean.shape != self.input_mean.shape or std.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Normalization needs finite feature-aligned mean and positive std')
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def forward(self, observations, hidden=None, mask=None):
        """Return world-frame Object21[B,T,21] and hidden[1,B,H].

        A supplied mask must be a nonempty right-padded sequence per batch row.
        Packing prevents padded steps from changing the final hidden state. Padding
        output is zero and must be excluded from loss and metrics.
        """
        if (observations.ndim != 3 or observations.shape[-1] != len(self.input_mean)
                or min(observations.shape[:2]) < 1 or not torch.isfinite(observations).all()):
            raise ValueError(f'Expected finite observations[B,T,{len(self.input_mean)}]')
        if hidden is not None and (hidden.shape != (1, observations.shape[0], self.hidden)
                                   or not torch.isfinite(hidden).all()):
            raise ValueError(f'hidden must be finite [1,B,{self.hidden}] or None')
        if mask is None:
            mask = torch.ones(observations.shape[:2], dtype=torch.bool, device=observations.device)
        if mask.dtype != torch.bool or mask.shape != observations.shape[:2]:
            raise ValueError('mask must be boolean [B,T]')
        lengths = mask.sum(1)
        expected = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
        if (lengths == 0).any() or not torch.equal(mask, expected):
            raise ValueError('mask must describe nonempty right-padded episodes')
        embedded = self.projection((observations - self.input_mean) / self.input_std)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        states, hidden = self.gru(packed, hidden)
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=observations.shape[1])
        # Decode only valid frames. A padded zero quaternion is structural padding,
        # not a missing observation; it must never pass through pose normalization.
        frozen = observations[..., :21][mask]
        poses = {'peg': frozen[..., :7], 'hole': frozen[..., 10:17]}
        for name, head in self.heads.items():
            start = 0 if name == 'peg' else 10
            correction = head(states[mask])
            position = correction[..., :3] * self.position_scale_m
            if self.position_mode == 'residual':
                position = position + frozen[..., start:start + 3]
            quaternion = quaternion_product(rotvec_quaternion(correction[..., 3:]),
                                              unit_quaternion(frozen[..., start + 3:start + 7]))
            poses[name] = torch.cat((position, unit_quaternion(quaternion)), dim=-1)
        composed = torch.cat((poses['peg'], frozen[..., 7:10], poses['hole'], frozen[..., 17:21]), dim=-1)
        refined = torch.cat((poses['peg'], geometric_endpoint(composed), poses['hole'], frozen[..., 17:21]), dim=-1)
        output = observations.new_zeros(observations.shape[:2] + (21,))
        output[mask] = refined
        return output, hidden

    def step(self, observation, hidden=None):
        """Advance one batched observation; reset by passing hidden=None."""
        if observation.ndim != 2:
            raise ValueError('step observations must have shape [B,D]')
        output, hidden = self(observation[:, None], hidden)
        return output[:, 0], hidden


class StreamingPoseEstimator:
    """Own hidden/TCP history for one episode; caller must reset at boundaries."""

    def __init__(self, model):
        self.model = model.eval()
        self.reset()

    def reset(self):
        self.hidden = self.previous_tcp = self.previous_quaternion = None

    @torch.inference_mode()
    def step(self, object21, quality25, tcp_world, tcp_quaternion=None, gripper_qpos=None,
             robot15=None, freshness2=None):
        """Return refined poses, consistent TCP-relative Object21 and optional State38."""
        if self.model.with_robot != (tcp_quaternion is not None and gripper_qpos is not None):
            raise ValueError('Streaming robot observations must match the trained feature contract')
        if (robot15 is None) != (freshness2 is None):
            raise ValueError('Supply robot15 and freshness2 together')
        inputs = pose_input_step(object21, quality25, tcp_world, self.previous_tcp,
                                 tcp_quaternion, self.previous_quaternion, gripper_qpos)
        world, self.hidden = self.model.step(inputs, self.hidden)
        self.previous_tcp = tcp_world.detach().clone()
        self.previous_quaternion = tcp_quaternion.detach().clone() if tcp_quaternion is not None else None
        relative = world.clone()
        relative[:, :3] -= tcp_world
        relative[:, 10:13] -= tcp_world
        result = {'world_object21': world, 'object21': relative,
                  'peg_pose_world': world[:, :7], 'hole_pose_world': world[:, 10:17],
                  'relative_hole_frame': hole_frame_vector(world[:, 7:10], world[:, 13:17]),
                  'peg_axis_in_hole_frame': peg_axis(relative_rotation(world[:, 3:7], world[:, 13:17]))}
        if robot15 is not None:
            result['state38'] = assemble_state38(robot15, relative, freshness2)
        return result
