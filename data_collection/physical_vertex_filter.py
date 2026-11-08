"""Select frames by distinct GT-associated detected peg and aperture corners.

This is a GT-assisted dataset filter. The association is finite-tolerance and
must not be interpreted as physical identity proven for every detection.
"""

import numpy as np

from data_collection.vertex_family_labels import SIGNS, pose_rotation, project_world


def peg_corner_ids(xyz_world, pixels, object_ids, ground_truth, observation,
                   coordinate_tolerance_m=0.003, projection_tolerance_px=2.0):
    """Match detected PEG points to a unique GT box corner; reject ambiguous matches."""
    xyz = np.asarray(xyz_world, dtype=np.float64)
    uv = np.asarray(pixels, dtype=np.float64)
    objects = np.asarray(object_ids)
    if xyz.shape != (len(objects), 3) or uv.shape != (len(objects), 2):
        raise ValueError("xyz_world and pixels must have shapes [N,3] and [N,2]")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Detected positions and pixels must be finite")
    if coordinate_tolerance_m <= 0 or projection_tolerance_px <= 0:
        raise ValueError("Coordinate and pixel tolerances must be positive")
    pose = np.asarray(ground_truth["peg_pose"], dtype=np.float64).reshape(7)
    half_size = np.asarray(ground_truth["peg_half_sizes"], dtype=np.float64).reshape(3)
    if not np.isfinite(half_size).all() or np.any(half_size <= 0):
        raise ValueError("peg_half_sizes must contain three positive finite values")
    rotation = pose_rotation(pose)
    local_corners = SIGNS * half_size
    world_corners = local_corners @ rotation.T + pose[:3]
    projected, camera = project_world(world_corners, observation["intrinsic_cv"], observation["extrinsic_cv"])
    ids = np.full(len(objects), -1, dtype=np.int64)
    for index in np.flatnonzero(objects == 0):
        local_point = (xyz[index] - pose[:3]) @ rotation
        candidates = np.flatnonzero(np.all(np.abs(local_corners - local_point) <= coordinate_tolerance_m, axis=1))
        if len(candidates) != 1:
            continue
        corner = candidates[0]
        if camera[corner, 2] > 0 and np.linalg.norm(projected[corner] - uv[index]) <= projection_tolerance_px:
            ids[index] = corner
    return ids


def physical_vertex_conditions(peg_ids, hole_family, hole_ids):
    """Count unique matches and reject peg corners all lying on one box face."""
    peg_ids = np.asarray(peg_ids, dtype=np.int64)
    family = np.asarray(hole_family, dtype=np.int64)
    hole_ids = np.asarray(hole_ids, dtype=np.int64)
    if family.shape != hole_ids.shape:
        raise ValueError("hole_family and hole_ids must have the same shape")
    if np.any((peg_ids < -1) | (peg_ids >= len(SIGNS))):
        raise ValueError("peg IDs must be -1 or between 0 and 7")
    peg_unique = np.unique(peg_ids[peg_ids >= 0])
    # Oracle RIM physical IDs are 8..15; exterior IDs 0..7 are never counted.
    rim_unique = np.unique(hole_ids[(family == 1) & (hole_ids >= 8) & (hole_ids < 16)])
    same_face = bool(len(peg_unique) and np.any(np.all(SIGNS[peg_unique] == SIGNS[peg_unique][0], axis=0)))
    peg_pass = len(peg_unique) >= 4 and not same_face
    hole_pass = len(rim_unique) >= 3
    return {
        "unique_peg_ids": peg_unique.tolist(), "unique_rim_ids": rim_unique.tolist(),
        "peg_single_face": same_face, "peg_pass": bool(peg_pass),
        "hole_pass": bool(hole_pass), "both_pass": bool(peg_pass and hole_pass),
    }
