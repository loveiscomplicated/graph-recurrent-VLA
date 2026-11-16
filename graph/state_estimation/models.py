"""Object21D state estimators sharing one spatial backbone and output head."""

import torch
from torch import nn

from graph.state_estimation.mamba import ObjectMamba
from graph.state_estimation.spatial import (
    Object21Head,
    SpatialBackbone,
    position_anchors,
)
from graph.state_estimation.temporal import (
    CausalAttention,
    episode_suffix,
    frame_times,
    temporal_vertices,
)


class SpatialEstimator(nn.Module):
    """Model A, using only the last frame without recurrent state."""

    spatial_frame_limit = 1

    def __init__(self, hidden=64, layers=2, dropout=0.0):
        super().__init__()
        self.backbone = SpatialBackbone(hidden, layers, dropout)
        self.head = Object21Head(hidden + 2, hidden)

    def spatial_nodes(self, graph):
        """Encode a frame and pass edge-conditioned spatial messages."""
        return self.backbone.propagate(self.backbone.encode(graph), graph)

    def spatial_objects(self, graph):
        """Extract two object-level embeddings from one frame."""
        hidden = self.spatial_nodes(graph)
        return self.backbone.pool(
            hidden, graph.object_id, graph.quality, graph.geometry
        )

    def forward(self, frames):
        """Predict Object21D at the latest frame of a batch of windows."""
        current = frames[-1]
        return self.head(self.spatial_objects(current), position_anchors(current))

    def forward_sequence(self, frames):
        """Compute comparable causal predictions at every supplied time."""
        return torch.stack(
            [self(frames[:end])["object21"] for end in range(1, len(frames) + 1)], dim=1
        )


class VertexTemporalEstimator(SpatialEstimator):
    """Shared B1/B2 readout and per-vertex attention capacity."""

    def __init__(self, hidden=64, layers=2, dropout=0.0, history=8, heads=4):
        super().__init__(hidden, layers, dropout)
        self.history = history
        self.temporal = CausalAttention(hidden, heads, dropout)

    def combine(self, current, hidden, temporal, binding, *, spatial_after):
        """Fuse verified current vertices and pool history-only vertices once."""
        device = hidden.device
        rows = binding.current.to(device)
        kinds = binding.kinds.to(device)
        batch = torch.arange(len(rows), device=device)[:, None].expand_as(rows)
        supported = current.geometry[batch, rows.clamp_min(0)]
        matched = (rows >= 0) & supported
        hidden = hidden.index_put(
            (batch[matched], rows[matched]), temporal[matched], accumulate=True
        )
        if spatial_after:
            hidden = self.backbone.propagate(hidden, current)
        extra = (kinds >= 0) & ~matched
        return self.backbone.pool(
            torch.cat((hidden, temporal), dim=1),
            torch.cat((current.object_id, kinds), dim=1),
            torch.cat((current.quality, current.quality.new_ones(rows.shape)), dim=1),
            torch.cat((current.geometry, extra), dim=1),
        )


class TemporalFirstEstimator(VertexTemporalEstimator):
    """Model B1, binding vertex histories before the current spatial pass."""

    def forward(self, frames):
        frames = frames[-self.history :]
        current = frames[-1]
        embeddings = [self.backbone.encode(frame) for frame in frames]
        temporal, binding = temporal_vertices(self.temporal, frames, embeddings)
        objects = self.combine(
            current, embeddings[-1], temporal, binding, spatial_after=True
        )
        return self.head(objects, position_anchors(current))


class SpatialFirstEstimator(VertexTemporalEstimator):
    """Model B2, binding vertices after each frame's spatial pass."""

    spatial_frame_limit = None

    def forward(self, frames):
        frames = frames[-self.history :]
        current = frames[-1]
        embeddings = [self.spatial_nodes(frame) for frame in frames]
        temporal, binding = temporal_vertices(self.temporal, frames, embeddings)
        objects = self.combine(
            current, embeddings[-1], temporal, binding, spatial_after=False
        )
        return self.head(objects, position_anchors(current))


class ObjectTemporalEstimator(SpatialEstimator):
    """Common spatial encoding and auxiliary head for object-level temporal models."""

    spatial_frame_limit = None

    def __init__(self, hidden, layers, dropout, history):
        super().__init__(hidden, layers, dropout)
        self.history = history
        self.object_projection = nn.Linear(hidden + 2, hidden)
        self.head = Object21Head(hidden, hidden)
        self.shape = nn.ModuleList((nn.Linear(hidden, 3), nn.Linear(hidden, 1)))

    def decode(self, objects, current):
        """Read physical state and static shape from the same temporal latents."""
        output = self.head(objects, position_anchors(current))
        raw_size = torch.cat(
            [head(objects[:, kind]) for kind, head in enumerate(self.shape)], dim=-1
        )
        output["shape"] = (
            torch.nn.functional.softplus(raw_size.float()) * self.head.size_scales
        )
        return output


class ObjectTransformerEstimator(ObjectTemporalEstimator):
    """Model C1, causal attention over per-object frame embeddings."""

    def __init__(self, hidden=64, layers=2, dropout=0.0, history=8, heads=4):
        super().__init__(hidden, layers, dropout, history)
        self.temporal = CausalAttention(hidden, heads, dropout)

    def forward(self, frames):
        frames = frames[-self.history :]
        objects = torch.stack([self.spatial_objects(frame) for frame in frames], dim=2)
        sequence = self.object_projection(objects).flatten(0, 1)
        starts = episode_suffix(frames)
        times = frame_times(frames, starts, sequence.device).repeat_interleave(2, dim=0)
        start_rows = torch.tensor(starts, device=sequence.device).repeat_interleave(2)
        visible = (
            torch.arange(len(frames), device=sequence.device) >= start_rows[:, None]
        )
        temporal = self.temporal(sequence, visible, times, start_rows)
        current = temporal[:, -1].reshape(objects.shape[0], 2, -1)
        return self.decode(current, frames[-1])


class ObjectMambaEstimator(ObjectTemporalEstimator):
    """Model C2, Mamba-1 over object latents with independent streaming caches."""

    def __init__(
        self,
        hidden=64,
        layers=2,
        dropout=0.0,
        history=8,
        tbptt_steps=0,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__(hidden, layers, dropout, history)
        self.temporal = ObjectMamba(hidden, tbptt_steps, d_state, d_conv, expand)

    @property
    def mixer(self):
        """Expose the underlying mixer for sequence/step verification."""
        return self.temporal.mixer

    def step(self, current, state=None):
        """Accept one frame and return its prediction and the next caller-owned cache."""
        objects = self.object_projection(self.spatial_objects(current))
        temporal, state = self.temporal(objects, current, state)
        return self.decode(temporal, current), state

    def forward(self, frames):
        """Predict with the same maximum number of frames as the attention models."""
        state = None
        for frame in frames[-self.history :]:
            output, state = self.step(frame, state)
        return output

    def forward_sequence(self, frames):
        """Replay the stream exactly as individual step calls, including episode reset."""
        outputs = []
        state = None
        for frame in frames:
            output, state = self.step(frame, state)
            outputs.append(output["object21"])
        return torch.stack(outputs, dim=1)


MODELS = {
    "spatial": SpatialEstimator,
    "temporal_first": TemporalFirstEstimator,
    "spatial_first": SpatialFirstEstimator,
    "object_transformer": ObjectTransformerEstimator,
    "object_mamba": ObjectMambaEstimator,
}


def build_model(
    name, *, hidden=64, layers=2, dropout=0.0, history=8, heads=4, tbptt_steps=0
):
    """Construct an estimator from matched settings shared by the comparison."""
    if name not in MODELS:
        raise ValueError(
            f"Unknown Object21D model {name!r}; choose from {tuple(MODELS)}"
        )
    if type(history) is not int or history < 1:
        raise ValueError("history must be a positive number of saved frames")
    if type(hidden) is not int or hidden < 8 or type(layers) is not int or layers < 1:
        raise ValueError("Expected hidden >= 8 and layers >= 1")
    settings = {"hidden": hidden, "layers": layers, "dropout": dropout}
    if name != "spatial":
        settings["history"] = history
    if name in ("temporal_first", "spatial_first", "object_transformer"):
        settings["heads"] = heads
    if name == "object_mamba":
        settings["tbptt_steps"] = tbptt_steps
    return MODELS[name](**settings)
