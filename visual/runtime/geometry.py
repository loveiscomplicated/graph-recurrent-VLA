"""Organized, observation-only NumPy geometry diagnostics (Task 01).

All arrays retain the source (v, u) image layout. This is an internal baseline,
not structural classification, sampling, or the final Vision--Graph interface.
No RGB values, object GT, erosion, outlier removal, or reconstruction are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from visual.runtime.rgbd import BOX, PEG, TrackedRGBDFrame


@dataclass(frozen=True)
class GeometryParameters:
    """Adjacent z jump > abs + rel * min(z1, z2), in metres.

    These are diagnostic baseline thresholds, not edge-classification settings.
    The same gate prevents a normal stencil from crossing a detected z jump.
    """

    depth_jump_abs_m: float = 0.01
    depth_jump_rel: float = 0.02

    def __post_init__(self) -> None:
        values = (self.depth_jump_abs_m, self.depth_jump_rel)
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("depth jump thresholds must be finite and nonnegative")


@dataclass(frozen=True)
class OrganizedGeometry:
    """CV-camera XYZ in metres; invalid depth has three NaNs, never a point.

    ``xyz[v, u]`` corresponds to source pixel centre ``(u+.5, v+.5)``.
    Finite background geometry is retained, but only ``observed_mask(id)`` may
    be used as that object's points. Overlapping tracker masks are preserved
    and marked ambiguous; those pixels are excluded from BOTH object supports.
    """

    xyz: np.ndarray
    valid_depth: np.ndarray
    object_masks: dict[int, np.ndarray]
    overlap_mask: np.ndarray

    def observed_mask(self, object_id: int) -> np.ndarray:
        return self.object_masks[object_id] & self.valid_depth & ~self.overlap_mask


@dataclass(frozen=True)
class GeometricFeatures:
    """Pixel-aligned diagnostics for one object, without semantic edge labels.

    Unavailable floating features are NaN and have explicit validity masks.
    ``depth_jump_m`` compares a supported centre to ANY finite 4-neighbour,
    including other labels/background; it measures observation discontinuity.
    Mask, invalid-depth, and image boundaries are separate diagnostic flags.
    ``mask_boundary`` can include invalid/ambiguous mask pixels: it is NOT a
    point selection. Geometry selection always requires ``valid_points``.
    Normal variation is the maximum angle (radians) to supported same-object
    4-neighbour normals; even a smooth curved surface can have a large value.
    """

    object_mask: np.ndarray
    valid_points: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray
    depth_jump_m: np.ndarray
    depth_pair_valid: np.ndarray
    depth_discontinuity: np.ndarray
    mask_boundary: np.ndarray
    depth_validity_boundary: np.ndarray
    image_boundary: np.ndarray
    normal_variation_rad: np.ndarray
    normal_variation_valid: np.ndarray
    normal_neighbor_count: np.ndarray


@dataclass(frozen=True)
class GeometryFrame:
    source_frame: int
    camera: str
    frame_idx: int | None
    timestamp: float | None
    geometry: OrganizedGeometry
    objects: dict[int, GeometricFeatures]
    timings_ms: dict[str, float]


def build_organized_geometry(frame: TrackedRGBDFrame) -> OrganizedGeometry:
    """Adapt pointcloud.backproject's pixel-centre equations to dense layout.

    Kept NumPy-only rather than importing the legacy plotting/SciPy module.
    No depth is filtered, filled, smoothed, or selected before feature analysis.
    """
    depth = np.asarray(frame.depth)
    if depth.ndim != 2 or not min(depth.shape) or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("depth must be a nonempty floating point (H,W) array in metres")
    intrinsic = np.asarray(frame.intrinsic, dtype=np.float64)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
        raise ValueError("intrinsic must be finite (3,3) with positive focal lengths")
    # The legacy equations assume the standard zero-skew CV pinhole intrinsic.
    if not np.allclose(intrinsic[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1]):
        raise ValueError("intrinsic must use the standard zero-skew CV pinhole form")
    if set(frame.masks) - {PEG, BOX}:
        raise ValueError("unknown output object ID; expected PEG=0 or BOX=1")
    masks = {}
    for object_id in (PEG, BOX):
        mask = np.asarray(frame.masks.get(object_id, np.zeros(depth.shape, dtype=bool)))
        if mask.shape != depth.shape or mask.dtype != np.bool_:
            raise ValueError("object masks must be boolean (H,W) arrays matching depth")
        masks[object_id] = mask.copy()
    valid = np.isfinite(depth) & (depth > 0)
    xyz = np.full((*depth.shape, 3), np.nan, dtype=np.float32)
    v, u = np.nonzero(valid)
    # Exactly the equations in visual/scripts/pointcloud.py::backproject.
    z = depth[v, u]
    xyz[v, u, 0] = (u + 0.5 - intrinsic[0, 2]) * z / intrinsic[0, 0]
    xyz[v, u, 1] = (v + 0.5 - intrinsic[1, 2]) * z / intrinsic[1, 1]
    xyz[v, u, 2] = z
    if not np.isfinite(xyz[valid]).all():
        raise ValueError("finite positive depth/intrinsic must yield representable float32 XYZ")
    return OrganizedGeometry(xyz, valid, masks, masks[PEG] & masks[BOX])


def _pairs():
    """Each in-image horizontal/vertical pair once; no wrap or padded neighbours."""
    yield (slice(None), slice(None, -1)), (slice(None), slice(1, None))
    yield (slice(None, -1), slice(None)), (slice(1, None), slice(None))


def _jump(z1: np.ndarray, z2: np.ndarray, parameters: GeometryParameters) -> np.ndarray:
    # Inputs have invalid z replaced by zero by callers, avoiding NaN/Inf math.
    return np.abs(z1 - z2) > (
        parameters.depth_jump_abs_m + parameters.depth_jump_rel * np.minimum(z1, z2)
    )


def estimate_surface_normals(
    geometry: OrganizedGeometry, object_id: int, parameters: GeometryParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Central cross product from a full, same-object 5-pixel (+) stencil.

    Centre and four neighbours must have unambiguous valid depth, and none of
    the four centre-to-neighbour pairs may exceed the z-jump gate. No one-sided
    fallback or extrapolation is performed at holes, mask/image borders, or
    tiny fragments. Unit normals point towards the camera (n dot XYZ <= 0).
    """
    xyz = geometry.xyz
    normals = np.full_like(xyz, np.nan)
    normal_valid = np.zeros(geometry.valid_depth.shape, dtype=bool)
    if min(normal_valid.shape) < 3:
        return normals, normal_valid
    support = geometry.observed_mask(object_id)
    centre = (slice(1, -1), slice(1, -1))
    neighbours = ((slice(1, -1), slice(None, -2)),
                  (slice(1, -1), slice(2, None)),
                  (slice(None, -2), slice(1, -1)),
                  (slice(2, None), slice(1, -1)))
    usable = support[centre].copy()
    z = np.where(geometry.valid_depth, xyz[..., 2], 0)
    for neighbour in neighbours:
        usable &= support[neighbour] & ~_jump(z[centre], z[neighbour], parameters)
    # Compute only complete stencils, so invalid XYZ never enters a cross product.
    v, u = np.nonzero(usable)
    v, u = v + 1, u + 1
    dx = xyz[v, u + 1].astype(np.float64) - xyz[v, u - 1]
    dy = xyz[v + 1, u].astype(np.float64) - xyz[v - 1, u]
    cross = np.cross(dx, dy)
    length = np.linalg.norm(cross, axis=1)
    good = np.isfinite(length) & (length > 0)
    v, u, cross = v[good], u[good], cross[good] / length[good, None]
    cross[np.einsum("ij,ij->i", cross, xyz[v, u]) > 0] *= -1
    normals[v, u] = cross.astype(np.float32)
    normal_valid[v, u] = True
    return normals, normal_valid


def compute_geometric_features(
    geometry: OrganizedGeometry, object_id: int, normals: np.ndarray,
    normal_valid: np.ndarray, parameters: GeometryParameters,
) -> GeometricFeatures:
    """Measure separate 4-neighbour boundary causes and local normal variation."""
    mask = geometry.object_masks[object_id]
    support = geometry.observed_mask(object_id)
    shape = mask.shape
    z = np.where(geometry.valid_depth, geometry.xyz[..., 2], 0)
    jump_m = np.zeros(shape, np.float32)
    pair_valid = np.zeros(shape, bool)
    discontinuity = np.zeros(shape, bool)
    mask_boundary = np.zeros(shape, bool)
    validity_boundary = np.zeros(shape, bool)
    variation = np.zeros(shape, np.float32)
    count = np.zeros(shape, np.uint8)
    for a, b in _pairs():
        both_depth = geometry.valid_depth[a] & geometry.valid_depth[b]
        dz = np.abs(z[a] - z[b])
        jumps = both_depth & _jump(z[a], z[b], parameters)
        both_normals = normal_valid[a] & normal_valid[b]
        angle = np.zeros(both_normals.shape, np.float32)
        n1, n2 = normals[a][both_normals], normals[b][both_normals]
        # atan2 is stable near zero, including identical rounded float32 normals.
        angle[both_normals] = np.arctan2(
            np.linalg.norm(np.cross(n1, n2), axis=1), np.einsum("ij,ij->i", n1, n2)
        )
        for p, q in ((a, b), (b, a)):
            available = support[p] & both_depth
            pair_valid[p] |= available
            np.maximum(jump_m[p], np.where(available, dz, 0), out=jump_m[p])
            discontinuity[p] |= support[p] & jumps
            mask_boundary[p] |= mask[p] & ~mask[q]
            validity_boundary[p] |= support[p] & ~geometry.valid_depth[q]
            np.maximum(variation[p], angle, out=variation[p])
            count[p] += both_normals
    jump_m[~pair_valid] = np.nan
    variation_valid = normal_valid & (count > 0)
    variation[~variation_valid] = np.nan
    image_boundary = np.zeros(shape, bool)
    image_boundary[[0, -1], :] = mask[[0, -1], :]
    image_boundary[:, [0, -1]] = mask[:, [0, -1]]
    return GeometricFeatures(
        object_mask=mask, valid_points=support, normals=normals, normal_valid=normal_valid,
        depth_jump_m=jump_m, depth_pair_valid=pair_valid, depth_discontinuity=discontinuity,
        mask_boundary=mask_boundary, depth_validity_boundary=validity_boundary,
        image_boundary=image_boundary, normal_variation_rad=variation,
        normal_variation_valid=variation_valid, normal_neighbor_count=count,
    )


def process_geometry(
    frame: TrackedRGBDFrame, parameters: GeometryParameters | None = None,
) -> GeometryFrame:
    """Process one current observation; no cache, final classification, or RGB features."""
    parameters = parameters or GeometryParameters()
    start = perf_counter()
    geometry = build_organized_geometry(frame)
    after_geometry = perf_counter()
    normals = {object_id: estimate_surface_normals(geometry, object_id, parameters)
               for object_id in geometry.object_masks}
    after_normals = perf_counter()
    features = {object_id: compute_geometric_features(
        geometry, object_id, *normals[object_id], parameters,
    ) for object_id in geometry.object_masks}
    end = perf_counter()
    timings = dict(geometry=(after_geometry - start) * 1000,
                   normals=(after_normals - after_geometry) * 1000,
                   discontinuity_features=(end - after_normals) * 1000,
                   total=(end - start) * 1000)
    return GeometryFrame(frame.source_frame, frame.camera, frame.frame_idx, frame.timestamp,
                         geometry, features, timings)
