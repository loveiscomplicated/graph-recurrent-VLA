"""Explicit per-batch cache for the repository's reference Mamba mixer."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from graph.temporal.insertion import ReferenceMamba


@dataclass(frozen=True)
class MambaState:
    """Caller-owned stream state. A batch row corresponds to one active episode."""

    episode_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    steps: tuple[int, ...]
    convolution: Tensor
    selective: Tensor


class ObjectMamba(nn.Module):
    """Use the same selective mixer independently for each object's latent stream."""

    def __init__(self, hidden, tbptt_steps, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if type(tbptt_steps) is not int or tbptt_steps < 0:
            raise ValueError("tbptt_steps must be a nonnegative integer")
        self.tbptt_steps = tbptt_steps
        self.norm = nn.LayerNorm(hidden)
        self.mixer = ReferenceMamba(hidden, d_state, d_conv, expand)

    def forward(self, objects, graph, state):
        """Advance one frame, clearing cache rows when their episode changes."""
        batch_size = len(objects)
        if state is None:
            convolution, selective = self.mixer.allocate_inference_cache(batch_size * 2)
            steps = (1,) * batch_size
        else:
            if len(state.steps) != batch_size:
                raise ValueError("Mamba stream batch size changed; start a new stream")
            resets = [
                old != new for old, new in zip(state.episode_ids, graph.episode_ids)
            ]
            for reset, previous, current in zip(
                resets, state.frame_indices, graph.frame_indices
            ):
                if not reset and current <= previous:
                    raise ValueError(
                        "Mamba stream frame indices must increase within an episode"
                    )
            steps = tuple(
                1 if reset else step + 1 for reset, step in zip(resets, state.steps)
            )
            detach = [
                bool(
                    self.training and self.tbptt_steps and step % self.tbptt_steps == 0
                )
                for step in state.steps
            ]
            reset_rows = torch.tensor(resets, device=objects.device).repeat_interleave(
                2
            )[:, None, None]
            detach_rows = torch.tensor(detach, device=objects.device).repeat_interleave(
                2
            )[:, None, None]

            def prepare(cache):
                cache = torch.where(detach_rows, cache.detach(), cache)
                return cache.masked_fill(reset_rows, 0.0)

            convolution, selective = (
                prepare(state.convolution),
                prepare(state.selective),
            )

        with torch.autocast(objects.device.type, enabled=False):
            tokens = self.norm(objects.float()).flatten(0, 1).unsqueeze(1)
            update, convolution, selective = self.mixer.step(
                tokens, convolution, selective
            )
        state = MambaState(
            graph.episode_ids, graph.frame_indices, steps, convolution, selective
        )
        return objects + update.reshape_as(objects), state
