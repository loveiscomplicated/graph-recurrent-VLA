"""Causal Mamba-1 object-state estimation with explicit reference/CUDA backends.

The reference mixer follows state-spaces/mamba v2.2.6 mamba_simple.py and its
selective_scan_ref equations. It is a selective SSM, including input-dependent
Delta/B/C and gated causal convolution; it is not the recurrent Action policy.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from graph.gated_mpnn import assemble_state38
from graph.pose_geometry import assemble_refined_object21, hole_frame_vector
from graph.temporal.pose_gru import MODES
from graph.temporal.pose_observations import (
    BASE_NAMES, FRESHNESS_NAMES, ROBOT_NAMES, observation_age_step, pose_input_step,
)


def feature_names(with_robot):
    """Use identical named observations for insertion GRU and Mamba experiments."""
    return BASE_NAMES + (ROBOT_NAMES if with_robot else ()) + FRESHNESS_NAMES


class ReferenceMamba(nn.Module):
    """Readable FP32 Mamba-1 mixer with persistent convolution and selective state.

    Parameter names/shapes match the official Mamba mixer for parity testing.
    Sequence projection/convolution is batched, but the reference selective scan
    is a Python loop. Its throughput does not represent the fused CUDA kernels.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank**-.5, self.dt_rank**-.5)
        # Official log-uniform time constants in [0.001, 0.1], inverse softplus bias.
        time_constants = torch.exp(torch.rand(self.d_inner) * math.log(100) + math.log(.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(time_constants + torch.log(-torch.expm1(-time_constants)))
        rates = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(rates.log())
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def allocate_inference_cache(self, batch_size, max_seqlen=None):
        """Allocate fixed-size cache, independent of episode length."""
        weight = self.in_proj.weight
        convolution = weight.new_zeros(batch_size, self.d_inner, self.d_conv)
        selective = weight.new_zeros(batch_size, self.d_inner, self.d_state)
        return convolution, selective

    def _selective_step(self, value, gate, state):
        """Apply the input-dependent discretized selective SSM recurrence."""
        delta, input_matrix, output_matrix = self.x_proj(value).split(
            (self.dt_rank, self.d_state, self.d_state), dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        transition = torch.exp(delta[..., None] * -self.A_log.exp())
        injection = delta[..., None] * input_matrix[:, None] * value[..., None]
        state = transition * state + injection
        output = (state * output_matrix[:, None]).sum(-1) + self.D * value
        return output * F.silu(gate), state

    def forward(self, sequence):
        """Causal parallel projections/convolution followed by the reference scan."""
        value, gate = self.in_proj(sequence).chunk(2, dim=-1)
        convolution = self.conv1d(F.pad(value.transpose(1, 2), (self.d_conv - 1, 0)))
        value = F.silu(convolution).transpose(1, 2)
        _, state = self.allocate_inference_cache(len(sequence))
        outputs = []
        for index in range(sequence.shape[1]):
            output, state = self._selective_step(value[:, index], gate[:, index], state)
            outputs.append(output)
        return self.out_proj(torch.stack(outputs, dim=1))

    def step(self, sequence, conv_state, ssm_state):
        """Advance one token using the same official cache layout [B,inner,W/N]."""
        value, gate = self.in_proj(sequence[:, 0]).chunk(2, dim=-1)
        convolution = torch.cat((conv_state[..., 1:], value[..., None]), dim=-1)
        value = (convolution * self.conv1d.weight[:, 0]).sum(-1) + self.conv1d.bias
        output, selective = self._selective_step(F.silu(value), gate, ssm_state)
        return self.out_proj(output)[:, None], convolution, selective


def build_mixer(backend, settings, layer_index):
    """Load official kernels only when explicitly requested; never silently fallback."""
    if backend == 'torch_reference':
        return ReferenceMamba(**settings)
    if backend != 'mamba_ssm':
        raise ValueError('model.backend must be torch_reference or mamba_ssm')
    try:
        from mamba_ssm import Mamba
    except ImportError as error:
        raise RuntimeError('mamba_ssm requires a compatible Linux CUDA installation; '
                           'see docs/gnn_training.md, or explicitly select torch_reference') from error
    return Mamba(**settings, layer_idx=layer_index, use_fast_path=True)


class TemporalInsertionEstimator(nn.Module):
    """Shared observation contract, pose heads and assembly for Mamba and GRU.

    Residual world-position correction is the default because the frozen Peg
    prediction already provides a useful spatial anchor. Rotation is a learned
    world-axis exponential-map residual left-composed with the GNN quaternion.
    No stationary-Hole or Peg-follows-TCP update rule is imposed.
    """

    def __init__(self, architecture='mamba', backend='torch_reference', mode='joint_peg_hole',
                 with_robot=True, d_model=128, n_layers=1, d_state=16, d_conv=4, expand=2,
                 head_hidden=64, position_mode='residual', position_scale_m=.01):
        super().__init__()
        if architecture not in ('mamba', 'gru') or mode not in MODES:
            raise ValueError('Expected architecture mamba/gru and a supported Peg/Hole mode')
        if type(with_robot) is not bool:
            raise ValueError('with_robot must be boolean')
        dimensions = (('d_model', d_model), ('n_layers', n_layers), ('d_state', d_state),
                      ('d_conv', d_conv), ('expand', expand), ('head_hidden', head_hidden))
        for name, value in dimensions:
            if type(value) is not int or value < 1:
                raise ValueError(f'model.{name} must be a positive integer')
        if position_mode not in ('direct', 'residual') or not 0 < position_scale_m < math.inf:
            raise ValueError('Expected direct/residual positions and finite positive position_scale_m')
        if backend not in ('torch_reference', 'mamba_ssm'):
            raise ValueError('model.backend must be torch_reference or mamba_ssm')
        self.architecture = architecture
        self.backend = backend if architecture == 'mamba' else 'torch_gru'
        self.mode = mode
        self.with_robot = with_robot
        self.position_mode = position_mode
        self.position_scale_m = position_scale_m
        self.d_model = d_model
        self.n_layers = n_layers
        width = len(feature_names(with_robot))
        self.register_buffer('input_mean', torch.zeros(width))
        self.register_buffer('input_std', torch.ones(width))
        self.projection = nn.Sequential(nn.Linear(width, d_model), nn.SiLU())
        if architecture == 'gru':
            self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        else:
            mixer_settings = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.mixers = nn.ModuleList([build_mixer(backend, mixer_settings, index)
                                         for index in range(n_layers)])
        self.heads = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 6),
        ) for name in MODES[mode]})

    def set_normalization(self, mean, std):
        """Store train-only statistics as checkpoint buffers."""
        if (mean.shape != self.input_mean.shape or std.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Expected feature-aligned finite mean and positive std')
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def _validate(self, observations, mask):
        """Reject empty, nonfinite or non-right-padded sequences before decoding."""
        if (observations.ndim != 3 or observations.shape[-1] != len(self.input_mean)
                or min(observations.shape[:2]) < 1 or not torch.isfinite(observations).all()):
            raise ValueError(f'Expected finite observations[B,T,{len(self.input_mean)}]')
        if self.backend == 'mamba_ssm' and observations.device.type != 'cuda':
            raise ValueError('Official mamba_ssm backend requires CUDA; select torch_reference for CPU')
        if mask is None:
            mask = torch.ones(observations.shape[:2], device=observations.device, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.shape != observations.shape[:2]:
            raise ValueError('Expected boolean mask[B,T]')
        lengths = mask.sum(1)
        expected = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
        if (lengths == 0).any() or not torch.equal(mask, expected):
            raise ValueError('Expected nonempty right-padded sequences')
        return mask

    def _decode(self, observations, representation, mask):
        """Decode valid states only, preserving structural zeros at padded frames."""
        corrections = {name: head(representation[mask]) for name, head in self.heads.items()}
        refined = assemble_refined_object21(observations[..., :21][mask], corrections,
                                            self.position_mode, self.position_scale_m)
        output = observations.new_zeros((*observations.shape[:2], 21))
        output[mask] = refined
        return output

    def forward_sequence(self, observations, mask=None):
        """Start with zero state; right-padding is causal and excluded from decoding."""
        mask = self._validate(observations, mask)
        representation = self.projection((observations - self.input_mean) / self.input_std)
        if self.architecture == 'gru':
            representation, _ = self.gru(representation)
        else:
            for norm, mixer in zip(self.norms, self.mixers):
                representation = representation + mixer(norm(representation))
        return self._decode(observations, representation, mask)

    def forward(self, observations, mask=None):
        return self.forward_sequence(observations, mask)

    def allocate_cache(self, batch_size):
        """Return independent GRU hidden or per-layer official-layout Mamba caches."""
        if self.architecture == 'gru':
            return self.input_mean.new_zeros(self.n_layers, batch_size, self.d_model)
        return [mixer.allocate_inference_cache(batch_size, max_seqlen=1) for mixer in self.mixers]

    def step(self, observations, cache):
        """Advance the actual recurrent/cache state; no history-window recomputation."""
        sequence = observations[:, None]
        mask = self._validate(sequence, None)
        representation = self.projection((sequence - self.input_mean) / self.input_std)
        if self.architecture == 'gru':
            representation, cache = self.gru(representation, cache)
        else:
            next_cache = []
            for norm, mixer, (convolution, selective) in zip(self.norms, self.mixers, cache):
                output, convolution, selective = mixer.step(norm(representation), convolution, selective)
                representation = representation + output
                next_cache.append((convolution, selective))
            cache = next_cache
        return self._decode(sequence, representation, mask)[:, 0], cache


class StreamingInsertionEstimator:
    """Manage independent cache and observed TCP/age history per episode ID.

    step accepts an observation dictionary containing only current model outputs
    and proprioception. reset(id) must be called at every real episode boundary.
    Interleaved IDs have independent state; release completed IDs with discard().
    """

    def __init__(self, model):
        self.model = model.eval()
        self.episodes = {}
        self.active_episode = None

    def reset(self, episode_id):
        self.active_episode = episode_id
        self.episodes[episode_id] = {'cache': None, 'tcp': None, 'quaternion': None, 'age': None}

    def discard(self, episode_id):
        """Release a completed episode's cache."""
        del self.episodes[episode_id]

    @torch.inference_mode()
    def forward_sequence(self, sequence, mask=None):
        """Evaluate prebuilt features from zero state, leaving live episode caches intact."""
        return self.model.forward_sequence(sequence, mask)

    @torch.inference_mode()
    def step(self, observation, episode_id=None):
        episode_id = self.active_episode if episode_id is None else episode_id
        if episode_id not in self.episodes:
            raise ValueError('Call reset(episode_id) before streaming a new episode')
        allowed = {'object21', 'quality25', 'tcp_world', 'tcp_quaternion', 'gripper_qpos',
                   'robot15', 'freshness2'}
        if set(observation) - allowed:
            raise ValueError('Streaming observation contains unknown fields; GT/labels are forbidden')
        robot15, freshness2 = observation.get('robot15'), observation.get('freshness2')
        if (robot15 is None) != (freshness2 is None):
            raise ValueError('Supply robot15 and freshness2 together for State38 assembly')
        state = self.episodes[episode_id]
        quaternion = observation.get('tcp_quaternion')
        gripper = observation.get('gripper_qpos')
        if self.model.with_robot != (quaternion is not None and gripper is not None):
            raise ValueError('Observed robot fields must match the checkpoint feature schema')
        quality = observation['quality25']
        inputs = pose_input_step(observation['object21'], quality, observation['tcp_world'],
                                 state['tcp'], quaternion, state['quaternion'], gripper)
        age = observation_age_step(quality, state['age'])
        inputs = torch.cat((inputs, age), dim=-1)
        if state['cache'] is None:
            state['cache'] = self.model.allocate_cache(len(inputs))
        world, state['cache'] = self.model.step(inputs, state['cache'])
        state.update(tcp=observation['tcp_world'].clone(),
                     quaternion=quaternion.clone() if quaternion is not None else None, age=age)
        relative = world.clone()
        relative[:, :3] -= observation['tcp_world']
        relative[:, 10:13] -= observation['tcp_world']
        result = {'world_object21': world, 'object21': relative,
                  'peg_pose_world': world[:, :7], 'hole_pose_world': world[:, 10:17],
                  'relative_hole_frame': hole_frame_vector(world[:, 7:10], world[:, 13:17]),
                  'observation_age_frames': age}
        if robot15 is not None:
            result['state38'] = assemble_state38(robot15, relative, freshness2)
        return result
