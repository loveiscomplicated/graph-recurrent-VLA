"""Sign-invariant TCP input for the shared corner predictor."""

from pathlib import Path

import torch
from torch import nn

from graph.temporal.causal_transformer import sinusoidal_positions
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor


INPUT_SCHEMA = "corner_history_tcp_rotation_6d_v1"
CHECKPOINT_FORMAT = "fixed_id_corner_rotation_6d_v1"


def encode_inputs(inputs, valid):
    """Replace raw wxyz at 7:11 by the first two rotation-matrix columns.

    The raw 13 values are previous XYZ, previous visibility, TCP XYZ/wxyz,
    and two finger positions. Padding is kept at zero.
    """
    if inputs.ndim != 3 or inputs.shape[-1] != 13:
        raise ValueError("inputs must have shape [B,T,13]")
    if valid.shape != inputs.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean [B,T] mask")
    quaternion = inputs[..., 7:11]
    norms = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if not torch.isfinite(inputs[valid]).all() or torch.any(norms[valid] < 1e-8):
        raise ValueError("Valid inputs must be finite and TCP quaternions nonzero")
    w, x, y, z = (quaternion / norms.clamp_min(1e-8)).unbind(-1)
    rotation = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z),
                            2 * (x*z - w*y), 2 * (x*y - w*z),
                            1 - 2 * (x*x + z*z), 2 * (y*z + w*x)), dim=-1)
    encoded = torch.cat((inputs[..., :7], rotation, inputs[..., 11:]), dim=-1)
    return torch.where(valid[..., None], encoded, torch.zeros_like(encoded))


class Rotation6DCornerPredictor(FixedIdCornerPredictor):
    """Accept raw histories and normalize sign-invariant 15D model features."""

    def __init__(self, **model_config):
        super().__init__(**model_config)
        width = self.projection.out_features
        self.projection = nn.Linear(15, width)
        self.input_mean = torch.zeros(15)
        self.input_std = torch.ones(15)

    def forward(self, inputs, context_valid):
        encoded = encode_inputs(inputs, context_valid)
        hidden = self.projection((encoded - self.input_mean) / self.input_std)
        if self.architecture == "mamba":
            for norm, mixer in zip(self.norms, self.mixers):
                hidden = hidden + mixer(norm(hidden))
        else:
            hidden = hidden + sinusoidal_positions(
                hidden.shape[1], hidden.shape[2], offset=0,
                device=hidden.device, dtype=hidden.dtype,
            )
            for block in self.blocks:
                hidden = block(hidden, padding_mask=~context_valid)
        return self.delta_head(self.output_norm(hidden)).float() * self.displacement_scale_m


def load_predictor(path, device="cpu"):
    """Load only the 6D checkpoint contract; reject legacy model weights."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("input_schema") != INPUT_SCHEMA:
        raise ValueError("Expected a fixed-ID corner rotation-6D checkpoint")
    model = Rotation6DCornerPredictor(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def load_tracker(path, distance_limit_m, device="cpu"):
    """Create a detector-only tracker from a 6D checkpoint and distance gate."""
    from graph.temporal.detected_corner_tracking import DetectedCornerTracker

    model, checkpoint = load_predictor(path, device)
    return DetectedCornerTracker(model, distance_limit_m, device), checkpoint
