"""Object21D coordinate and scale conventions for state estimation."""

import numpy as np
import torch

from graph.gated_mpnn import assemble_state38
from graph.utils.object21_targets import object21_from_gt

__all__ = ["State21Scale", "assemble_state38", "camera_to_world", "object21_from_gt"]

# Fixed physical scales avoid fitting any statistics on held-out episodes.
COORDINATE_SCALE_M = 0.05
SIZE_SCALES_M = (0.04, 0.01, 0.01, 0.02)


def camera_to_world(camera_xyz, extrinsic):
    """Invert the stored OpenCV world-to-camera transform for metre-valued rows."""
    camera_xyz = np.asarray(camera_xyz)
    extrinsic = np.asarray(extrinsic)
    if camera_xyz.shape[-1:] != (3,) or extrinsic.shape != (3, 4):
        raise ValueError("Expected camera xyz [...,3] and OpenCV extrinsic [3,4]")
    if not np.isfinite(camera_xyz).all() or not np.isfinite(extrinsic).all():
        raise ValueError("Camera coordinates and extrinsic must be finite")
    rotation = extrinsic[:, :3]
    if (
        not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
        or np.linalg.det(rotation) < 0
    ):
        raise ValueError("OpenCV extrinsic must contain a proper rotation")
    return (camera_xyz - extrinsic[:, 3]) @ rotation


class State21Scale:
    """Normalize metre-valued fields and preserve wxyz quaternion components."""

    def __init__(self, coordinate_m=COORDINATE_SCALE_M, sizes_m=SIZE_SCALES_M):
        self.units = (
            (coordinate_m,) * 3
            + (1.0,) * 4
            + (coordinate_m,) * 6
            + (1.0,) * 4
            + tuple(sizes_m)
        )
        if len(self.units) != 21 or not all(
            np.isfinite(unit) and unit > 0 for unit in self.units
        ):
            raise ValueError(
                "Object21 scaling requires positive finite coordinate and four size scales"
            )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Convert Object21D in physical units to dimensionless fields."""
        return state / self._units(state)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """Restore physical Object21D units."""
        return state * self._units(state)

    def _units(self, state):
        if state.shape[-1:] != (21,) or not state.is_floating_point():
            raise ValueError("Expected floating-point Object21D [...,21]")
        return state.new_tensor(self.units)
