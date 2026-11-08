"""One causal temporal predictor whose weights are shared by every corner."""

import torch
from torch import nn

from graph.temporal.causal_transformer import CausalTransformerBlock, sinusoidal_positions


class FixedIdCornerPredictor(nn.Module):
    """Predict displacement from the last detection using independent histories.

    Inputs contain 13 observed values per frame. IDs, GT geometry and current
    detection coordinates are absent. Each batch item is one corner sequence;
    right padding is ignored, and there is no external state or track manager.
    """

    def __init__(self, architecture="mamba", width=128, layers=2, heads=4,
                 d_state=16, d_conv=4, expand=2, displacement_scale_m=0.01):
        super().__init__()
        if architecture not in ("mamba", "transformer"):
            raise ValueError("architecture must be mamba or transformer")
        if width < 4 or layers < 1 or heads < 1 or width % heads:
            raise ValueError("Need positive layers/heads and width divisible by heads")
        if displacement_scale_m <= 0:
            raise ValueError("displacement_scale_m must be positive")
        self.architecture = architecture
        self.displacement_scale_m = displacement_scale_m
        self.register_buffer("input_mean", torch.zeros(13))
        self.register_buffer("input_std", torch.ones(13))
        self.projection = nn.Linear(13, width)

        # Backbones must not change the common projection/head initialization.
        with torch.random.fork_rng(devices=[]):
            if architecture == "mamba":
                try:
                    from mamba_ssm import Mamba
                except ImportError as error:
                    raise RuntimeError("Mamba requires the installed mamba_ssm CUDA runtime") from error
                self.mixers = nn.ModuleList([
                    Mamba(d_model=width, d_state=d_state, d_conv=d_conv, expand=expand)
                    for _ in range(layers)
                ])
                self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
            else:
                self.blocks = nn.ModuleList([
                    CausalTransformerBlock(width, heads, 2 * width, dropout=0.0)
                    for _ in range(layers)
                ])
        self.output_norm = nn.LayerNorm(width)
        self.delta_head = nn.Linear(width, 3)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, inputs, context_valid):
        """Return [B,T,3] world-axis displacement in metres, without future access."""
        if inputs.ndim != 3 or inputs.shape[-1] != 13:
            raise ValueError("inputs must have shape [B,T,13]")
        if context_valid.shape != inputs.shape[:2] or context_valid.dtype != torch.bool:
            raise ValueError("context_valid must be a boolean [B,T] right-padding mask")
        hidden = self.projection((inputs - self.input_mean) / self.input_std)
        if self.architecture == "mamba":
            # Causality makes padded suffixes irrelevant to every valid prefix.
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
