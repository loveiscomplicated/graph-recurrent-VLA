"""Causal windows and verified per-vertex associations."""

from dataclasses import dataclass

import torch
from torch import nn

from graph.temporal.causal_transformer import (
    CausalTransformerBlock,
    sinusoidal_positions,
)


def episode_suffix(frames):
    """Select the trailing same-episode run independently for each batch row."""
    suffixes = []
    for sample, episode_id in enumerate(frames[-1].episode_ids):
        start = len(frames) - 1
        while start > 0 and frames[start - 1].episode_ids[sample] == episode_id:
            if (
                frames[start - 1].frame_indices[sample]
                >= frames[start].frame_indices[sample]
            ):
                raise ValueError("Episode frame indices must increase")
            start -= 1
        suffixes.append(start)
    return suffixes


def frame_times(frames, starts, device):
    """Return source frame offsets, preserving gaps in sampled observations."""
    indices = torch.tensor([frame.frame_indices for frame in frames], device=device).T
    first = indices.gather(1, torch.tensor(starts, device=device)[:, None])
    return (indices - first).clamp_min(0)


@dataclass
class VertexBinding:
    keys: list[list[tuple[int, int]]]
    rows: torch.Tensor
    current: torch.Tensor
    kinds: torch.Tensor


def bind_vertices(frames):
    """Bind verified visible IDs within each episode, retaining history-only keys."""
    starts = episode_suffix(frames)
    keys, rows, current_rows = [], [], []
    for sample, start in enumerate(starts):
        indices_by_time = [
            frame.observed_vertex_rows[sample] if index >= start else {}
            for index, frame in enumerate(frames)
        ]
        bound = sorted(set().union(*indices_by_time))
        keys.append(bound)
        rows.append(
            [[lookup.get(key, -1) for lookup in indices_by_time] for key in bound]
        )
        current_rows.append(
            [frames[-1].vertex_rows[sample].get(key, -1) for key in bound]
        )

    count = max(map(len, keys))
    shape = len(keys), count
    rows_tensor = torch.full((*shape, len(frames)), -1, dtype=torch.long)
    current = torch.full(shape, -1, dtype=torch.long)
    kinds = torch.full(shape, -1, dtype=torch.long)
    for sample, bound in enumerate(keys):
        if bound:
            rows_tensor[sample, : len(bound)] = torch.tensor(rows[sample])
            current[sample, : len(bound)] = torch.tensor(current_rows[sample])
            kinds[sample, : len(bound)] = torch.tensor([key[0] for key in bound])
    return VertexBinding(keys, rows_tensor, current, kinds)


class CausalAttention(nn.Module):
    """One shared causal attention block with a safe key for fully hidden prefixes."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.block = CausalTransformerBlock(width, heads, width * 2, dropout)
        self.missing = nn.Parameter(torch.zeros(width))
        self.clock = nn.Linear(1, width, bias=False)

    def forward(self, sequence, visible, times, starts):
        """Mask unavailable keys, giving missing queries access to visible history."""
        batch, steps, width = sequence.shape
        token = self.missing.to(sequence.dtype)
        sequence = torch.where(visible[..., None], sequence, token)
        positions = sinusoidal_positions(
            steps, width, offset=0, device=sequence.device, dtype=sequence.dtype
        )
        order = torch.arange(steps, device=sequence.device) - starts[:, None]
        positions = positions[order.clamp_min(0)]
        clock = self.clock(times[..., None].to(sequence.dtype).log1p())
        sequence = sequence + positions + clock
        tokens = torch.cat((token.expand(batch, 1, width), sequence), dim=1)
        padding = torch.cat((visible.new_zeros(batch, 1), ~visible), dim=1)
        return self.block(tokens, padding)[:, 1:]


def temporal_vertices(attention, frames, embeddings):
    """Gather aligned histories, returning current embeddings and bound rows."""
    binding = bind_vertices(frames)
    batch, count, steps = binding.rows.shape
    device = embeddings[-1].device
    if count == 0:
        return embeddings[-1].new_zeros(batch, 0, embeddings[-1].shape[-1]), binding
    rows = binding.rows.to(device)
    samples = torch.arange(batch, device=device)[:, None]
    gathered = torch.stack(
        [
            hidden[samples, rows[:, :, time].clamp_min(0)]
            for time, hidden in enumerate(embeddings)
        ],
        dim=2,
    )
    suffixes = episode_suffix(frames)
    times = frame_times(frames, suffixes, device)
    starts = torch.tensor(suffixes, device=device).repeat_interleave(count)
    times = times[:, None, :].expand(batch, count, steps).flatten(0, 1)
    history = attention(
        gathered.flatten(0, 1), (rows >= 0).flatten(0, 1), times, starts
    )
    return history[:, -1].reshape(batch, count, -1), binding
