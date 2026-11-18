"""Object-Delaunay GNN followed by a causal Transformer corner tracker."""

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from graph.state_estimation.spatial import SpatialBackbone
from graph.state_estimation.codec import COORDINATE_SCALE_M
from graph.temporal.causal_transformer import CausalTransformerBlock, sinusoidal_positions


CORNER_OBJECT_IDS = tuple([0] * 8 + [1] * 16)


class SpatialTemporalCornerTracker(nn.Module):
    """Forecast corner tracks from spatial context and past associated detections.

    Current detector coordinates enter the spatial GNN, but a track forecast at
    time ``t`` receives only its track-specific embedding from ``t-1``.  The
    pooled current-object context is available at ``t`` so visible TAPNext++
    surface motion can inform a corner while that corner is occluded.
    """

    def __init__(
        self,
        hidden=64,
        spatial_layers=2,
        temporal_layers=2,
        heads=4,
        dropout=0.0,
        association_dim=32,
        position_scale_m=0.01,
        coordinate_frame="world",
    ):
        super().__init__()
        if (
            type(hidden) is not int
            or hidden < 8
            or type(spatial_layers) is not int
            or spatial_layers < 1
            or type(temporal_layers) is not int
            or temporal_layers < 1
            or type(heads) is not int
            or heads < 1
            or hidden % heads
            or type(association_dim) is not int
            or association_dim < 2
            or position_scale_m <= 0
        ):
            raise ValueError("Invalid spatial-temporal tracker dimensions")
        self.hidden = hidden
        self.association_dim = association_dim
        self.position_scale_m = float(position_scale_m)
        if coordinate_frame not in ("world", "tcp_relative"):
            raise ValueError("coordinate_frame must be 'world' or 'tcp_relative'")
        self.coordinate_frame = coordinate_frame
        self.spatial = SpatialBackbone(hidden, spatial_layers, dropout)
        self.object_projection = nn.Linear(hidden + 2, hidden)
        self.missing_history = nn.Parameter(torch.zeros(hidden))
        self.observation_embedding = nn.Embedding(2, hidden)
        self.age_projection = nn.Linear(1, hidden, bias=False)
        self.temporal_blocks = nn.ModuleList(
            CausalTransformerBlock(hidden, heads, 2 * hidden, dropout)
            for _ in range(temporal_layers)
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.displacement = nn.Linear(hidden, 3)
        self.log_variance = nn.Linear(hidden, 1)
        self.track_embedding = nn.Linear(hidden, association_dim)
        self.detection_embedding = nn.Linear(hidden, association_dim)
        nn.init.zeros_(self.displacement.weight)
        nn.init.zeros_(self.displacement.bias)
        nn.init.zeros_(self.log_variance.weight)
        nn.init.zeros_(self.log_variance.bias)
        self.register_buffer(
            "corner_object_ids", torch.tensor(CORNER_OBJECT_IDS, dtype=torch.long)
        )

    def encode_spatial(self, graph):
        """Return node embeddings, projected object context and association keys."""
        if self.coordinate_frame == "world":
            # Missing-corner targets use world displacement. Preserve current
            # world motion even when a grasped Peg moves together with TCP.
            coordinates = graph.position / COORDINATE_SCALE_M
            features = torch.cat((coordinates, graph.features[..., 3:]), dim=-1)
            graph = replace(graph, features=features * graph.valid[..., None])
        node_hidden = self.spatial.propagate(self.spatial.encode(graph), graph)
        objects = self.spatial.pool(
            node_hidden, graph.object_id, graph.quality, graph.geometry
        )
        object_context = self.object_projection(objects)
        detection_keys = F.normalize(self.detection_embedding(node_hidden), dim=-1)
        return node_hidden, object_context, detection_keys

    def make_token(self, previous_hidden, previous_observed, age, object_context):
        """Fuse prior node evidence with current same-object spatial context."""
        if previous_hidden.shape[-1] != self.hidden:
            raise ValueError("previous_hidden has the wrong feature width")
        previous_observed = previous_observed.bool()
        history = torch.where(
            previous_observed[..., None],
            previous_hidden,
            self.missing_history.to(previous_hidden.dtype),
        )
        observed_feature = self.observation_embedding(previous_observed.long())
        age_feature = self.age_projection(torch.log1p(age.float())[..., None])
        return history + object_context + observed_feature + age_feature

    def temporal_forward(self, tokens, valid):
        """Apply full-history causal attention to right-padded track tokens."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden:
            raise ValueError("tokens must have shape [tracks,time,hidden]")
        if valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [tracks,time] mask")
        if not valid[:, 0].all():
            raise ValueError("Every temporal track must contain an initial token")
        hidden = tokens + sinusoidal_positions(
            tokens.shape[1], self.hidden, offset=0, device=tokens.device, dtype=tokens.dtype
        )
        for block in self.temporal_blocks:
            hidden = block(hidden, padding_mask=~valid)
        return self.output_norm(hidden)

    def decode_temporal(self, hidden, last_observed):
        """Decode world-axis displacement, uncertainty and association embedding."""
        delta = self.displacement(hidden).float() * self.position_scale_m
        log_variance = self.log_variance(hidden).float().squeeze(-1).clamp(-8.0, 4.0)
        association = F.normalize(self.track_embedding(hidden), dim=-1)
        return {
            "position": last_observed + delta,
            "delta": delta,
            "log_variance": log_variance,
            "track_embedding": association,
        }

    def forward(self, graph, rows, frame_mask, last_observed, age):
        """Forecast all 24 supervised slots without using their current rows."""
        if rows.ndim != 3 or rows.shape[-1] != len(CORNER_OBJECT_IDS):
            raise ValueError("rows must have shape [B,T,24]")
        batch, steps, slots = rows.shape
        if frame_mask.shape != (batch, steps) or last_observed.shape != (
            batch,
            steps,
            slots,
            3,
        ):
            raise ValueError("Frame masks and last-observed coordinates are misaligned")
        if age.shape != rows.shape:
            raise ValueError("age must have shape [B,T,24]")

        node_hidden, object_context, detection_keys = self.encode_spatial(graph)
        if node_hidden.shape[0] != batch * steps:
            raise ValueError("Graph frame count must equal B*T")
        width = node_hidden.shape[1]
        node_hidden = node_hidden.reshape(batch, steps, width, self.hidden)
        object_context = object_context.reshape(batch, steps, 2, self.hidden)
        detection_keys = detection_keys.reshape(
            batch, steps, width, self.association_dim
        )

        previous_rows = torch.full_like(rows, -1)
        previous_rows[:, 1:] = rows[:, :-1]
        previous_observed = previous_rows >= 0
        samples = torch.arange(batch, device=rows.device)[:, None, None]
        times = torch.arange(steps, device=rows.device)[None, :, None]
        previous_times = (times - 1).clamp_min(0)
        previous_hidden = node_hidden[
            samples, previous_times, previous_rows.clamp_min(0)
        ]
        previous_observed[:, 0] = False

        object_ids = self.corner_object_ids[None, None, :].expand(batch, steps, -1)
        context = object_context[samples, times, object_ids]
        tokens = self.make_token(previous_hidden, previous_observed, age, context)

        # A runtime track does not exist before its first detection. Left-align
        # every supervised history at that first sighting so training and online
        # sinusoidal positions and available context are identical.
        current_visible = (rows >= 0) & frame_mask[..., None]
        has_track = current_visible.any(dim=1)
        first_seen = current_visible.float().argmax(dim=1)
        first_seen = torch.where(has_track, first_seen, torch.zeros_like(first_seen))
        relative_time = torch.arange(steps, device=rows.device)[None, None, :]
        source_time = (first_seen[..., None] + relative_time).clamp_max(steps - 1)
        by_track = tokens.transpose(1, 2)
        shifted = torch.gather(
            by_track,
            2,
            source_time[..., None].expand(batch, slots, steps, self.hidden),
        )
        episode_length = frame_mask.sum(dim=1)[:, None]
        track_length = torch.where(
            has_track, episode_length - first_seen, torch.ones_like(first_seen)
        )
        valid = relative_time < track_length[..., None]
        shifted_temporal = self.temporal_forward(
            shifted.reshape(batch * slots, steps, self.hidden),
            valid.reshape(batch * slots, steps),
        ).reshape(batch, slots, steps, self.hidden)
        original_relative = (
            torch.arange(steps, device=rows.device)[None, :, None] - first_seen[:, None, :]
        )
        gather_relative = original_relative.clamp(0, steps - 1).transpose(1, 2)
        temporal = torch.gather(
            shifted_temporal,
            2,
            gather_relative[..., None].expand(batch, slots, steps, self.hidden),
        ).transpose(1, 2)
        output = self.decode_temporal(temporal, last_observed)
        output.update(
            node_hidden=node_hidden,
            detection_embedding=detection_keys,
            object_context=object_context,
            previous_observed=previous_observed,
        )
        return output


def contextual_assignment(
    prediction,
    track_kinds,
    track_embedding,
    detections,
    detection_kinds,
    detection_embedding,
    *,
    distance_limit_m,
    embedding_weight_m,
):
    """Maximize valid matches, then minimize position-plus-context cost."""
    prediction = np.asarray(prediction, dtype=np.float64)
    detections = np.asarray(detections, dtype=np.float64)
    tracks, points = len(prediction), len(detections)
    if distance_limit_m <= 0 or embedding_weight_m < 0:
        raise ValueError("Association distance must be positive and embedding weight nonnegative")
    if not tracks or not points:
        return []
    distance = np.linalg.norm(prediction[:, None] - detections[None], axis=-1)
    similarity = np.asarray(track_embedding) @ np.asarray(detection_embedding).T
    contextual_cost = distance + embedding_weight_m * (1.0 - similarity.clip(-1, 1))
    allowed = (
        np.asarray(track_kinds)[:, None] == np.asarray(detection_kinds)[None]
    ) & (distance <= distance_limit_m)
    maximum_pair_cost = distance_limit_m + 2 * embedding_weight_m
    dummy_cost = (tracks + 1) * max(maximum_pair_cost, 1e-8)
    cost = np.full((tracks, points + tracks), dummy_cost, dtype=np.float64)
    cost[:, :points] = np.where(allowed, contextual_cost, dummy_cost * (tracks + 2))
    row, column = linear_sum_assignment(cost)
    return [
        (int(track), int(point))
        for track, point in zip(row, column)
        if point < points and allowed[track, point]
    ]


@dataclass
class ContextTrack:
    """One detector-only lifecycle track with causal model tokens."""

    track_id: int
    kind: int
    last_observed: torch.Tensor
    previous_hidden: torch.Tensor
    previous_observed: bool = True
    age: int = 0
    tokens: list[torch.Tensor] = field(default_factory=list)
    layer_history: list[torch.Tensor] = field(default_factory=list)


class OnlineSpatialTemporalTracker:
    """Run the learned tracker without canonical IDs or reference geometry."""

    def __init__(
        self,
        model,
        *,
        distance_limit_m=0.03,
        embedding_weight_m=0.01,
    ):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.distance_limit_m = float(distance_limit_m)
        self.embedding_weight_m = float(embedding_weight_m)
        self.tracks = []

    def _advance_temporal(self, tracks):
        """Evaluate one new query against every stored causal layer state.

        Each layer retains all past normalized states, so attention still uses
        the complete track history. Only past queries avoid recomputation.
        """
        lengths = torch.tensor(
            [len(track.tokens) for track in tracks], device=self.device
        )
        tokens = torch.stack([track.tokens[-1] for track in tracks])
        positions = sinusoidal_positions(
            max(len(track.tokens) for track in tracks), self.model.hidden,
            offset=0, device=self.device, dtype=tokens.dtype,
        )
        hidden = tokens + positions[lengths - 1]
        histories = [[] for _ in tracks]
        for layer_index, block in enumerate(self.model.temporal_blocks):
            normalized = block.attention_norm(hidden)
            keys = []
            for index, track in enumerate(tracks):
                previous = (
                    track.layer_history[layer_index]
                    if track.layer_history
                    else normalized.new_empty((0, self.model.hidden))
                )
                history = torch.cat((previous, normalized[index : index + 1]))
                histories[index].append(history)
                keys.append(history)
            padded_keys = pad_sequence(keys, batch_first=True)
            padding = torch.arange(padded_keys.shape[1], device=self.device)[None] >= lengths[:, None]
            update, _ = block.attention(
                normalized[:, None], padded_keys, padded_keys,
                key_padding_mask=padding, need_weights=False,
            )
            hidden = hidden + block.residual_dropout(update[:, 0])
            hidden = hidden + block.residual_dropout(block.feedforward(block.feedforward_norm(hidden)))
        for track, history in zip(tracks, histories, strict=True):
            track.layer_history = history
        return self.model.output_norm(hidden)

    def _forecast(self, object_context):
        if not self.tracks:
            empty = torch.empty((0,), device=self.device)
            return empty.reshape(0, 3), empty.reshape(0, self.model.association_dim), empty
        for track in self.tracks:
            context = object_context[0, track.kind]
            token = self.model.make_token(
                track.previous_hidden[None],
                torch.tensor([track.previous_observed], device=self.device),
                # Training defines age at the current query frame. A point
                # observed in the immediately preceding frame therefore has
                # age one, while the stored post-update state is zero.
                torch.tensor([track.age + 1], device=self.device),
                context[None],
            )[0]
            track.tokens.append(token)
        current = self._advance_temporal(self.tracks)
        anchors = torch.stack([track.last_observed for track in self.tracks])
        decoded = self.model.decode_temporal(current, anchors)
        uncertainty = self.model.position_scale_m * torch.exp(0.5 * decoded["log_variance"])
        return decoded["position"], decoded["track_embedding"], uncertainty

    @torch.inference_mode()
    def step(self, graph):
        """Encode one observed graph, associate corners and retain missing tracks."""
        if graph.features.shape[0] != 1:
            raise ValueError("Online tracking expects exactly one frame graph")
        node_hidden, object_context, detection_keys = self.model.encode_spatial(graph)
        corner_mask = graph.valid[0] & (graph.features[0, :, 5] > 0.5)
        corner_rows = torch.nonzero(corner_mask, as_tuple=True)[0]
        detections = graph.position[0, corner_rows]
        kinds = graph.object_id[0, corner_rows]
        keys = detection_keys[0, corner_rows]
        prediction, track_keys, uncertainty = self._forecast(object_context)
        matches = contextual_assignment(
            prediction.cpu().numpy(),
            [track.kind for track in self.tracks],
            track_keys.cpu().numpy(),
            detections.cpu().numpy(),
            kinds.cpu().numpy(),
            keys.cpu().numpy(),
            distance_limit_m=self.distance_limit_m,
            embedding_weight_m=self.embedding_weight_m,
        )

        output_positions = [value.clone() for value in prediction]
        output_observed = [False] * len(self.tracks)
        output_uncertainty = uncertainty.tolist()
        detection_tracks = torch.full(
            (len(corner_rows),), -1, dtype=torch.long, device=self.device
        )
        for track in self.tracks:
            track.previous_observed = False
            track.age += 1
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            position = detections[detection_index]
            track.last_observed = position
            track.previous_hidden = node_hidden[0, corner_rows[detection_index]]
            track.previous_observed = True
            track.age = 0
            output_positions[track_index] = position
            output_observed[track_index] = True
            output_uncertainty[track_index] = 0.0
            detection_tracks[detection_index] = track.track_id

        new_tracks = []
        for detection_index in torch.nonzero(detection_tracks < 0, as_tuple=True)[0].tolist():
            kind = int(kinds[detection_index])
            position = detections[detection_index]
            first_token = self.model.make_token(
                node_hidden.new_zeros((1, self.model.hidden)),
                torch.zeros(1, dtype=torch.bool, device=self.device),
                torch.zeros(1, device=self.device),
                object_context[:, kind],
            )[0]
            track = ContextTrack(
                track_id=len(self.tracks),
                kind=kind,
                last_observed=position,
                previous_hidden=node_hidden[0, corner_rows[detection_index]],
                tokens=[first_token],
            )
            self.tracks.append(track)
            new_tracks.append(track)
            detection_tracks[detection_index] = track.track_id
            output_positions.append(position)
            output_observed.append(True)
            output_uncertainty.append(0.0)
        if new_tracks:
            self._advance_temporal(new_tracks)

        positions = torch.stack(output_positions) if output_positions else detections.new_empty((0, 3))
        return {
            "positions": positions,
            "forecast_positions": prediction,
            "observed": torch.tensor(output_observed, dtype=torch.bool, device=self.device),
            "uncertainty": torch.tensor(output_uncertainty, dtype=torch.float32, device=self.device),
            "age": torch.tensor([track.age for track in self.tracks], dtype=torch.long, device=self.device),
            "kinds": torch.tensor([track.kind for track in self.tracks], dtype=torch.long, device=self.device),
            "track_ids": torch.tensor([track.track_id for track in self.tracks], dtype=torch.long, device=self.device),
            "detection_rows": corner_rows,
            "detection_tracks": detection_tracks,
            "matches": len(matches),
            "new_tracks": len(corner_rows) - len(matches),
        }


def load_tracker_checkpoint(path, expected_sha256, device):
    """Load and freeze a B2-3 tracker checkpoint with an optional hash pin."""
    from data_collection.episode_dataset import file_sha256

    path = Path(path).expanduser().resolve()
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("Spatial-temporal tracker checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "spatial_temporal_corner_tracker_v1":
        raise ValueError(f"Unsupported spatial-temporal tracker checkpoint: {path}")
    model_settings = dict(checkpoint["config"]["model"])
    # Checkpoints saved before this explicit field encoded TCP-relative XYZ.
    model_settings.setdefault("coordinate_frame", "tcp_relative")
    model = SpatialTemporalCornerTracker(**model_settings)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint
