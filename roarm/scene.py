"""Fixed RoArm bench geometry and base-frame obstacle validation."""

import numpy as np

TABLE_HALF_SIZE = [0.65, 0.65, 0.02]
TABLE_POSITION = [0, 0, 0.48]
MOUNT_POSITION = [0, 0, 0.5]


def validate_obstacles(obstacles):
    """Copy static boxes defined by base-frame center and positive half sizes.

    For example: [{"center": [0.31, 0, 0.25], "half_size": [0.025, 0.035, 0.025]}].
    Both vectors contain three values in metres.
    """
    validated = []
    for index, obstacle in enumerate(obstacles):
        try:
            center = np.asarray(obstacle["center"], dtype=float)
            half_size = np.asarray(obstacle["half_size"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Obstacle {index} requires center and half_size xyz vectors"
            ) from exc
        if (
            center.shape != (3,)
            or half_size.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(half_size).all()
            or (half_size <= 0).any()
        ):
            raise ValueError(
                f"Obstacle {index}: center and half_size must be finite xyz "
                "vectors, with positive half sizes in metres"
            )
        validated.append({"center": center.copy(), "half_size": half_size.copy()})
    return validated
