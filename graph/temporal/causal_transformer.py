"""Causal Transformer blocks with full-episode streaming history."""

import math

import torch
from torch import nn


def sinusoidal_positions(length, width, *, offset, device, dtype):
    """Return deterministic positions; offsets preserve streaming/full-sequence parity."""
    position = torch.arange(offset, offset + length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.0) / width))
    angles = position * frequency
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, :width // 2].cos()
    return encoding.to(dtype=dtype)


class CausalTransformerBlock(nn.Module):
    """Pre-normalized attention and feed-forward block, without future access."""

    def __init__(self, width, heads, feedforward_width, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, sequence, padding_mask):
        """Apply a causal mask and exclude right-padding from attention keys."""
        length = sequence.shape[1]
        causal_mask = torch.ones(length, length, device=sequence.device, dtype=torch.bool).triu(1)
        normalized = self.attention_norm(sequence)
        update, _ = self.attention(normalized, normalized, normalized, attn_mask=causal_mask,
                                   key_padding_mask=padding_mask, need_weights=False)
        sequence = sequence + self.residual_dropout(update)
        return sequence + self.residual_dropout(self.feedforward(self.feedforward_norm(sequence)))

    def step(self, sequence, history):
        """Attend to stored past normalized states plus the current frame."""
        normalized = self.attention_norm(sequence)
        history = torch.cat((history, normalized), dim=1)
        update, _ = self.attention(normalized, history, history, need_weights=False)
        sequence = sequence + self.residual_dropout(update)
        sequence = sequence + self.residual_dropout(self.feedforward(self.feedforward_norm(sequence)))
        return sequence, history
