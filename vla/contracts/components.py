"""Ports inferred from the working live path, not new model architectures."""
from typing import Any, Protocol


class GraphBuilder(Protocol):
    coordinate_frame: str
    last_raw_masks: Any
    robot_filter_backend: str
    last_robot_schedule: str | None
    last_wrist_gate_backend: str | None

    def build(self, cameras, data, surfaces, masks, *, mask_transform=None): ...
    def from_h5(self, file, timestep, episode, *, tensor_images=True): ...
    def reset(self): ...
    def close(self): ...


class GraphEncoder(Protocol):
    output_dim: int

    def forward(self, graphs, trace=None):
        """Return per-graph feature and presence tensors on the input device."""
        ...


class Controller(Protocol):
    input_dim: int
    action_dim: int
    state_batch_dim: int

    def forward(self, sequence, state=None):
        """Return action sequence and a tensor-tree state, or None."""
        ...
