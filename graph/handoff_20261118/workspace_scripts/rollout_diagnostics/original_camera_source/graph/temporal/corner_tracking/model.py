"""Shared per-track motion decoder with interchangeable temporal backbones."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from graph.pose_geometry import conjugate, quaternion_product, quaternion_rotvec
from graph.temporal.causal_transformer import CausalTransformerBlock, sinusoidal_positions
from graph.temporal.insertion import build_mixer


RECORD_WIDTH = 42
QUERY_WIDTH = 24
CONTEXT_WIDTH = 16


@dataclass
class MotionPrediction:
    """World-axis displacement and diagonal variance, in metres and metres²."""

    delta: torch.Tensor
    variance: torch.Tensor


def differentiable_mamba_step(mixer, value, convolution, selective):
    """Functional Mamba-1 update for autograd; official step caches mutate in place.

    Reuse the selected mixer's actual parameters. The training path intentionally
    uses Torch operations, including on CUDA, rather than inference-only updates.
    """
    projected, gate = mixer.in_proj(value[:, 0]).chunk(2, dim=-1)
    convolution = torch.cat((convolution[..., 1:], projected[..., None]), dim=-1)
    projected = (convolution * mixer.conv1d.weight[:, 0]).sum(-1) + mixer.conv1d.bias
    projected = nn.functional.silu(projected)
    delta, input_matrix, output_matrix = mixer.x_proj(projected).split(
        (mixer.dt_rank, mixer.d_state, mixer.d_state), dim=-1)
    delta = nn.functional.softplus(mixer.dt_proj(delta))
    transition = torch.exp(delta[..., None] * -mixer.A_log.exp())
    selective = transition * selective + delta[..., None] * input_matrix[:, None] * projected[..., None]
    output = (selective * output_matrix[:, None]).sum(-1) + mixer.D * projected
    output = output * nn.functional.silu(gate)
    return mixer.out_proj(output)[:, None], convolution, selective


def rotation_change(current, previous):
    """Use relative rotations rather than subtracting quaternion components."""
    return quaternion_rotvec(quaternion_product(current, conjugate(previous)))


def observed_rotation_change(current, previous):
    """Compute nondifferentiable observed robot features before the GPU transfer."""
    previous = previous.copy()
    previous[..., 1:] *= -1
    left_w, left_v = current[..., :1], current[..., 1:]
    right_w, right_v = previous[..., :1], previous[..., 1:]
    relative = np.concatenate((left_w * right_w - (left_v * right_v).sum(-1, keepdims=True),
                               left_w * right_v + right_w * left_v + np.cross(left_v, right_v)), axis=-1)
    relative /= np.linalg.norm(relative, axis=-1, keepdims=True)
    relative = np.where(relative[..., :1] < 0, -relative, relative)
    magnitude = np.linalg.norm(relative[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(magnitude, relative[..., :1])
    factor = np.where(magnitude > 1e-7, angle / np.maximum(magnitude, 1e-8), 2.0)
    return relative[..., 1:] * factor


class CornerMotionModel(nn.Module):
    """Share weights across tracks while retaining a separate cache per track.

    predict() reads history through t-1. record() commits observation t only
    after association. Track IDs are cache keys, never numerical model features.
    """

    def __init__(self, architecture="mamba", backend="torch_reference", d_model=128,
                 layers=1, heads=4, d_state=16, d_conv=4, expand=2,
                 coordinate_scale_m=0.05, age_scale_frames=32.0):
        super().__init__()
        if architecture not in ("mamba", "transformer"):
            raise ValueError("architecture must be mamba or transformer")
        if (type(d_model) is not int or d_model < 8 or type(layers) is not int
                or layers < 1 or type(heads) is not int or heads < 1
                or d_model % heads or coordinate_scale_m <= 0 or age_scale_frames <= 0):
            raise ValueError("Invalid motion width/layers/heads or nonpositive feature scale")
        self.architecture = architecture
        self.backend = backend
        self.width = d_model
        self.coordinate_scale_m = coordinate_scale_m
        self.age_scale_frames = age_scale_frames
        self._mamba_step = differentiable_mamba_step
        self.context_encoder = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, CONTEXT_WIDTH))
        self.record_encoder = nn.Sequential(nn.Linear(RECORD_WIDTH, d_model), nn.SiLU())
        # Backbone draws must not change shared encoder/head initialization.
        with torch.random.fork_rng(devices=[]):
            if architecture == "mamba":
                settings = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                self.mixers = nn.ModuleList([build_mixer(backend, settings, i) for i in range(layers)])
                self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
            else:
                self.blocks = nn.ModuleList([
                    CausalTransformerBlock(d_model, heads, 2 * d_model, 0.0)
                    for _ in range(layers)
                ])
        self.motion_head = nn.Sequential(
            nn.Linear(d_model + QUERY_WIDTH, d_model), nn.SiLU(), nn.Linear(d_model, 6),
        )
        # Begin with last-observation hold and finite, nonzero uncertainty.
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        with torch.no_grad():
            self.motion_head[-1].bias[3:].fill_(-1.0)
        self._predict_forward = self._predict_values
        self._context_forward = self._context_values

    def _predict_values(self, hidden, query):
        """Tensor-only prediction region, including uncertainty decoding."""
        raw = self.motion_head(torch.cat((hidden, query), dim=-1))
        delta = raw[:, :3] * self.coordinate_scale_m
        variance = (-8.0 + 12.0 * raw[:, 3:].sigmoid()).exp() * self.coordinate_scale_m ** 2
        return delta, variance

    def _context_values(self, inputs, membership):
        """Pool all object types without dynamic boolean indexing on CUDA."""
        values = self.context_encoder(inputs)
        return membership.T @ values / membership.sum(0).clamp_min(1)[:, None]

    @property
    def device(self):
        return next(self.parameters()).device

    def tensor(self, value):
        """Convert observations, never supervision, onto the model device."""
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def new_memory(self):
        """Allocate fresh state at birth; anchor refreshes do not reset it."""
        if self.architecture == "mamba":
            caches = [mixer.allocate_inference_cache(1, max_seqlen=1) for mixer in self.mixers]
        else:
            caches = [torch.empty(1, 0, self.width, device=self.device) for _ in self.blocks]
        return dict(caches=caches, hidden=torch.zeros(self.width, device=self.device), length=0)

    def predict(self, memory, anchor, anchor_robot, robot, age, kind, quality, quality_valid):
        """Predict total displacement from the current fixed observed anchor."""
        robot = self.tensor(robot)
        anchor_robot = self.tensor(anchor_robot)
        robot_features = torch.cat((robot[:3] / self.coordinate_scale_m,
                                    robot[3:7], robot[7:9] / 0.04))
        query = torch.cat((
            (self.tensor(anchor) - robot[:3]) / self.coordinate_scale_m,
            robot_features,
            (robot[:3] - anchor_robot[:3]) / self.coordinate_scale_m,
            rotation_change(robot[3:7], anchor_robot[3:7]),
            self.tensor([age / self.age_scale_frames]),
            nn.functional.one_hot(torch.tensor(kind, device=self.device), 3).float(),
            self.tensor([quality, float(quality_valid)]),
        ))
        raw = self.motion_head(torch.cat((memory["hidden"], query)))
        # Bound log variance in normalized units to avoid numerical singularities.
        log_variance = -8.0 + 12.0 * raw[3:].sigmoid()
        return MotionPrediction(raw[:3] * self.coordinate_scale_m,
                                log_variance.exp() * self.coordinate_scale_m ** 2)

    def predict_many(self, tracks, robot, frame):
        """Decode all live track queries together, preserving per-track gradients."""
        if not tracks:
            return []
        count = len(tracks)
        current = np.broadcast_to(robot, (count, 9))
        anchors = np.stack([track.anchor for track in tracks])
        anchor_robots = np.stack([track.anchor_robot for track in tracks])
        kinds = np.eye(3, dtype=np.float32)[[track.kind for track in tracks]]
        ages = np.asarray([frame - track.anchor_frame for track in tracks], dtype=np.float32)[:, None]
        qualities = np.asarray([[track.quality, float(track.quality_valid)] for track in tracks], dtype=np.float32)
        robot_features = np.concatenate((current[:, :3] / self.coordinate_scale_m,
                                         current[:, 3:7], current[:, 7:9] / 0.04), axis=-1)
        query = np.concatenate((
            (anchors - current[:, :3]) / self.coordinate_scale_m,
            robot_features,
            (current[:, :3] - anchor_robots[:, :3]) / self.coordinate_scale_m,
            observed_rotation_change(current[:, 3:7], anchor_robots[:, 3:7]),
            ages / self.age_scale_frames,
            kinds, qualities,
        ), axis=-1)
        hidden = torch.stack([track.memory["hidden"] for track in tracks])
        delta, variance = self._predict_forward(hidden, self.tensor(query))
        return [MotionPrediction(delta[index], variance[index]) for index in range(count)]

    def contexts(self, observation):
        """Permutation-invariant same-object context, after synthetic removal."""
        inputs = np.concatenate(((observation.xyz - observation.robot[:3]) / self.coordinate_scale_m,
                                 observation.confidence[:, None],
                                 observation.confidence_valid[:, None]), axis=-1).astype(np.float32)
        membership = np.eye(3, dtype=np.float32)[observation.kind]
        pooled = self._context_forward(self.tensor(inputs), self.tensor(membership))
        return {kind: pooled[kind] for kind in range(3)}

    def record(self, memory, robot, previous_robot, kind, age, context,
               xyz=None, quality=0.0, quality_valid=False):
        """Commit exactly one actual-observation or missing token per live frame."""
        robot = self.tensor(robot)
        previous = robot if previous_robot is None else self.tensor(previous_robot)
        robot_features = torch.cat((
            robot[:3] / self.coordinate_scale_m, robot[3:7], robot[7:9] / 0.04,
            (robot[:3] - previous[:3]) / self.coordinate_scale_m,
            rotation_change(robot[3:7], previous[3:7]), self.tensor([1.0]),
        ))
        position = (torch.zeros(3, device=self.device) if xyz is None else
                    (self.tensor(xyz) - robot[:3]) / self.coordinate_scale_m)
        record = torch.cat((position, self.tensor([float(xyz is not None), quality,
                                                  float(quality_valid)]),
                            nn.functional.one_hot(torch.tensor(kind, device=self.device), 3).float(),
                            self.tensor([age / self.age_scale_frames]), robot_features, context))
        value = self.record_encoder(record)[None, None]
        next_caches = []
        if self.architecture == "mamba":
            for norm, mixer, cache in zip(self.norms, self.mixers, memory["caches"]):
                update, convolution, selective = differentiable_mamba_step(mixer, norm(value), *cache)
                value = value + update
                next_caches.append((convolution, selective))
        else:
            value = value + sinusoidal_positions(1, self.width, offset=memory["length"],
                                                device=self.device, dtype=value.dtype)[None]
            for block, history in zip(self.blocks, memory["caches"]):
                value, history = block.step(value, history)
                next_caches.append(history)
        return dict(caches=next_caches, hidden=value[0, 0], length=memory["length"] + 1)

    def record_many(self, records, robot, previous_robot):
        """Update track caches in batches; Transformer groups equal history lengths."""
        if not records:
            return []
        count = len(records)
        current = robot
        previous = current if previous_robot is None else previous_robot
        robot_features = np.concatenate((
            current[:3] / self.coordinate_scale_m, current[3:7], current[7:9] / 0.04,
            (current[:3] - previous[:3]) / self.coordinate_scale_m,
            observed_rotation_change(current[3:7], previous[3:7]), np.ones(1, dtype=np.float32),
        ))
        robot_features = np.broadcast_to(robot_features, (count, len(robot_features)))
        positions = np.stack([np.zeros(3, dtype=np.float32) if row["xyz"] is None else row["xyz"]
                              for row in records])
        present = np.asarray([float(row["xyz"] is not None) for row in records], dtype=np.float32)[:, None]
        positions = (positions - current[:3]) / self.coordinate_scale_m * present
        qualities = np.asarray([[row["quality"], float(row["quality_valid"])] for row in records], dtype=np.float32)
        kinds = np.eye(3, dtype=np.float32)[[row["kind"] for row in records]]
        ages = np.asarray([row["age"] for row in records], dtype=np.float32)[:, None] / self.age_scale_frames
        contexts = torch.stack([row["context"] for row in records])
        observed_features = np.concatenate((positions, present, qualities, kinds, ages, robot_features), axis=-1)
        tokens = torch.cat((self.tensor(observed_features), contexts), dim=-1)
        encoded = self.record_encoder(tokens)[:, None, :]
        groups = {}
        for index, row in enumerate(records):
            key = row["memory"]["length"] if self.architecture == "transformer" else 0
            groups.setdefault(key, []).append(index)
        updated = [None] * count
        for length, indices in groups.items():
            value = encoded[indices]
            memories = [records[index]["memory"] for index in indices]
            batch_caches = []
            if self.architecture == "mamba":
                for layer, (norm, mixer) in enumerate(zip(self.norms, self.mixers)):
                    convolution = torch.cat([memory["caches"][layer][0] for memory in memories])
                    selective = torch.cat([memory["caches"][layer][1] for memory in memories])
                    change, convolution, selective = self._mamba_step(
                        mixer, norm(value), convolution, selective)
                    value = value + change
                    batch_caches.append((convolution, selective))
            else:
                value = value + sinusoidal_positions(1, self.width, offset=length,
                                                      device=self.device, dtype=value.dtype)[None]
                for layer, block in enumerate(self.blocks):
                    history = torch.cat([memory["caches"][layer] for memory in memories])
                    value, history = block.step(value, history)
                    batch_caches.append(history)
            for local, index in enumerate(indices):
                if self.architecture == "mamba":
                    caches = [(convolution[local:local + 1], selective[local:local + 1])
                              for convolution, selective in batch_caches]
                else:
                    caches = [history[local:local + 1] for history in batch_caches]
                updated[index] = dict(caches=caches, hidden=value[local, 0],
                                      length=memories[local]["length"] + 1)
        return updated


class ConstantVelocityMotion(nn.Module):
    """Nonlearned bootstrap tracker used for calibration and comparison."""

    def __init__(self, mode="constant_velocity", device="cpu"):
        super().__init__()
        if mode not in ("constant_velocity", "hold"):
            raise ValueError("Baseline mode must be constant_velocity or hold")
        self.mode = mode
        self.register_buffer("location", torch.empty(0, device=device))

    @property
    def device(self):
        return self.location.device

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def new_memory(self):
        return dict(length=0, observations=[])

    def contexts(self, observation):
        return {kind: torch.zeros(CONTEXT_WIDTH, device=self.device) for kind in range(3)}

    def predict(self, memory, anchor, anchor_robot, robot, age, kind, quality, quality_valid):
        observations = memory["observations"]
        velocity = torch.zeros(3, device=self.device)
        if self.mode == "constant_velocity" and len(observations) >= 2:
            first, second = observations[-2:]
            velocity = (self.tensor(second[1]) - self.tensor(first[1])) / (second[0] - first[0])
        return MotionPrediction(velocity * age, torch.full((3,), 1e-6, device=self.device))

    def record(self, memory, robot, previous_robot, kind, age, context,
               xyz=None, quality=0.0, quality_valid=False):
        observations = memory["observations"]
        if xyz is not None:
            observations = (observations + [(memory["length"], xyz.copy())])[-2:]
        return dict(length=memory["length"] + 1, observations=observations)
