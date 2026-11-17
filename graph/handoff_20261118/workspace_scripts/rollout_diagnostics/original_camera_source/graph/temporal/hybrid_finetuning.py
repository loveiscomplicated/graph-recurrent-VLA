"""Peg-only adaptation of the existing H1 assembly, with frozen branch inputs."""

import torch
from torch import nn

from graph.temporal.hybrid import assemble_hybrid
from graph.pose_geometry import assemble_refined_object21


class HybridPegFineTuner(nn.Module):
    """Expose only the existing independent six-output Peg head to optimization.

    Cached representations come from the unchanged, causal Mamba backbone.
    Gate positions and the original Mamba Hole pose are detached observations,
    never supervised targets. No adapter or new inference parameters are added.
    """

    def __init__(self, mamba, gate):
        super().__init__()
        if mamba.architecture != 'mamba' or set(mamba.heads) != {'peg', 'hole'}:
            raise ValueError('H1 fine-tuning requires a joint Mamba with separate Peg/Hole heads')
        self.mamba = mamba
        self.gate = gate
        self.enable_peg_head()

    def enable_peg_head(self):
        """Restore the freeze policy after an inference helper freezes all branches."""
        self.requires_grad_(False)
        self.mamba.heads['peg'].requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        """Keep frozen modules in evaluation mode even while the Peg head trains."""
        super().train(False)
        self.mamba.heads['peg'].train(mode)
        return self

    def forward(self, representation, frozen_world, mamba_world, gate_world):
        """Decode valid frames only; tensors use metres and world wxyz poses."""
        correction = self.mamba.heads['peg'](representation.detach())
        peg = assemble_refined_object21(
            frozen_world.detach(), {'peg': correction},
            self.mamba.position_mode, self.mamba.position_scale_m)
        adapted = torch.cat((peg[..., :7], mamba_world.detach()[..., 7:]), dim=-1)
        return assemble_hybrid(adapted, gate_world.detach(), frozen_world.detach())


def snapshot_state(model):
    """Keep an independent CPU copy, including normalization buffers."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def parameter_changes(before, model):
    """Reject any change outside heads.peg; return every tensor's exact delta."""
    after = model.state_dict()
    if set(before) != set(after):
        raise AssertionError('Model state keys changed during Peg-only fine-tuning')
    changes = {}
    for name, old in before.items():
        current = after[name].detach().cpu()
        changed = not torch.equal(old, current)
        if changed and not name.startswith('heads.peg.'):
            raise AssertionError(f'Frozen Mamba tensor changed: {name}')
        changes[name] = {'changed': changed, 'max_absolute_delta': float((old - current).abs().max())}
    return changes
