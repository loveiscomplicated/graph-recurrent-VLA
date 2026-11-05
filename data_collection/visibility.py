"""GT/depth first-frame visibility, extracted from the historical validated gate.

The default gate requires both peg ends and the expert grasp band. GT is used
only for dataset selection. Depth input remains signed int16 millimetres.
"""

from dataclasses import dataclass
import numpy as np
import sapien
import trimesh
from scipy.spatial import ConvexHull
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
)

QUATERNION_NORM_ATOL = 1e-5


def pose_matrix(pose):
    """World-from-EE pose [x,y,z,qw,qx,qy,qz], with no identity fallback."""
    p = np.asarray(pose, np.float64)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError("finite source observation TCP pose [7] required")
    norm = np.linalg.norm(p[3:])
    if abs(norm - 1) > QUATERNION_NORM_ATOL:
        raise ValueError("TCP wxyz quaternion is not unit")
    w, x, y, z = p[3:] / norm
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    t[:3, 3] = p[:3]
    return t


def validate_transform(transform):
    t = np.asarray(transform)
    if (
        t.shape != (4, 4)
        or not np.issubdtype(t.dtype, np.floating)
        or not np.isfinite(t).all()
    ):
        raise ValueError("finite floating [4,4] transform required")
    if not np.array_equal(t[3], [0, 0, 0, 1]):
        raise ValueError("invalid homogeneous row")
    r = t[:3, :3].astype(np.float64)
    atol = 1e-5
    if np.max(np.abs(r.T @ r - np.eye(3))) > atol or abs(np.linalg.det(r) - 1) > atol:
        raise ValueError("calibration rotation is not rigid within recorded precision")
    return t.astype(np.float64)


def transform(points, matrix):
    matrix = validate_transform(np.asarray(matrix, dtype=float))
    return np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]


def validate_camera_metadata(camera):
    """Small host identity/calibration checks shared by CPU and tensor images."""
    k = np.asarray(camera.intrinsic)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or k[0, 1] != 0
        or k[1, 0] != 0
    ):
        raise ValueError("CV intrinsic")
    cv = np.vstack((camera.extrinsic_cv, [0, 0, 0, 1]))
    validate_transform(cv.astype(float))
    gl = validate_transform(np.asarray(camera.cam2world_gl, float))
    # Independent conventions stored by the installed camera API.
    if not np.allclose(
        np.linalg.inv(cv), gl @ np.diag([1, -1, -1, 1]), atol=2e-5, rtol=0
    ):
        raise ValueError("CV/GL camera calibration disagreement")
    pose_matrix(camera.tcp_pose)
    if not np.isfinite(camera.timestamp):
        raise ValueError("timestamp")


@dataclass(frozen=True)
class CameraObservation:
    episode: str
    timestep: int
    camera: str
    timestamp: float
    rgb: np.ndarray
    depth_raw_mm: np.ndarray
    intrinsic: np.ndarray
    extrinsic_cv: np.ndarray
    cam2world_gl: np.ndarray
    tcp_pose: np.ndarray

    def validate(self):
        if not self.episode or not self.camera or self.timestep < 0:
            raise ValueError("observation identity")
        h, w = self.depth_raw_mm.shape
        if self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValueError("RGB/depth alignment")
        if self.depth_raw_mm.dtype != np.int16:
            raise ValueError("original signed int16 millimetres required")
        validate_camera_metadata(self)
        return self

    @property
    def depth(self):
        # Preserve overflow and nonpositive values in the source; never unwrap
        # -32768 into fabricated positive depth.
        z = self.depth_raw_mm.astype(np.float64) / 1000
        return np.where(self.depth_raw_mm > 0, z, np.nan)

    @property
    def world_from_camera(self):
        # Match the existing recorded/live G0 CV calibration contract exactly.
        return np.linalg.inv(np.vstack((self.extrinsic_cv, [0, 0, 0, 1]))).astype(
            np.float32
        )


def _box_depth(camera, local_vertices, world_from_peg):
    """Ray/convex half-space clipping; depth is CV z in metres.

    Nonunit rays have camera z=1, so the ray parameter itself is metric depth.
    Entry intersections model the camera-facing surface of the GT peg box.
    """
    h, w = camera.depth_raw_mm.shape
    v, u = np.indices((h, w))
    k = camera.intrinsic
    rays = np.stack(
        ((u + 0.5 - k[0, 2]) / k[0, 0], (v + 0.5 - k[1, 2]) / k[1, 1], np.ones((h, w))),
        -1,
    ).reshape(-1, 3)
    wc = camera.world_from_camera
    direction = rays @ wc[:3, :3].T
    origin = wc[:3, 3]
    zbuffer = np.full(h * w, np.inf)
    for vertices in [transform(local_vertices, world_from_peg)]:
        vc = transform(vertices, np.linalg.inv(wc))
        if np.all(vc[:, 2] <= 0):
            continue
        if np.all(vc[:, 2] > 0):
            uv = (
                vc[:, :2] / vc[:, 2, None] * np.array([k[0, 0], k[1, 1]])
                + k[:2, 2]
                - 0.5
            )
            lo = np.maximum(np.floor(uv.min(0)).astype(int) - 1, [0, 0])
            hi = np.minimum(np.ceil(uv.max(0)).astype(int) + 1, [w - 1, h - 1])
            if np.any(hi < lo):
                continue
            vv, uu = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
            rows = (vv * w + uu).ravel()
        else:
            rows = np.arange(h * w)
        equations = ConvexHull(vertices).equations
        normals, offset = equations[:, :3], equations[:, 3]
        numerator = -(normals @ origin + offset)
        denominator = direction[rows] @ normals.T
        parallel = np.abs(denominator) < 1e-12
        crossing = np.divide(
            numerator, denominator, out=np.zeros_like(denominator), where=~parallel
        )
        lower = np.max(np.where(denominator < -1e-12, crossing, -np.inf), axis=1)
        upper = np.min(np.where(denominator > 1e-12, crossing, np.inf), axis=1)
        outside_parallel = np.any(parallel & (numerator < 0), axis=1)
        entry = np.maximum(lower, 0)
        intersects = (upper >= entry) & (upper > 0) & ~outside_parallel
        hit = intersects & (entry < zbuffer[rows])
        zbuffer[rows[hit]] = entry[hit]
    return zbuffer.reshape(h, w)


SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


def grasp_target(peg_pose, half, tcp_pose):
    """Same installed OBB helper, Panda grasp frame and offset as ScriptedExpert.reset."""
    T = pose_matrix(peg_pose)
    obb = trimesh.creation.box(
        extents=2 * np.asarray(half), transform=T
    ).bounding_box_oriented
    hint = pose_matrix(tcp_pose)[:3, 1]
    info = compute_grasp_info_by_obb(
        obb, approaching=np.array([0, 0, -1]), target_closing=hint, depth=0.025
    )
    g = Panda.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
    g = g * sapien.Pose([-max(0.05, float(half[0]) / 2 + 0.01), 0, 0])
    world = np.asarray(g.p, float)
    return world, (world - T[:3, 3]) @ T[:3, :3]


def projection(camera, points, near=0.01, epsilon=0.0001):
    E = np.asarray(camera.extrinsic_cv, float)
    xyz = np.asarray(points) @ E[:, :3].T + E[:, 3]
    uv = np.full((len(xyz), 2), np.nan)
    front = xyz[:, 2] > near
    uv[front] = (
        xyz[front, :2]
        / xyz[front, 2:]
        * np.array([camera.intrinsic[0, 0], camera.intrinsic[1, 1]])
        + camera.intrinsic[:2, 2]
    )
    h, w = camera.depth_raw_mm.shape
    inside = (
        front
        & (uv[:, 0] >= -epsilon)
        & (uv[:, 0] <= w + epsilon)
        & (uv[:, 1] >= -epsilon)
        & (uv[:, 1] <= h + epsilon)
    )
    return xyz, uv, inside


def evaluate(camera, peg_pose, half, tcp_pose, *, pad_width=0.0175, tolerance=0.0015):
    camera.validate()
    half = np.asarray(half, float)
    T = pose_matrix(peg_pose)
    local = SIGNS * half
    world = transform(local, T)
    xyz, uv, inside = projection(camera, world)
    target, g_local = grasp_target(peg_pose, half, tcp_pose)
    # Longitudinal footprint of the actual rubber tip collision box local-x width.
    gh = pad_width / 2
    gx = float(g_local[0])
    slab = SIGNS * np.array([gh, half[1], half[2]]) + np.array([gx, 0, 0])
    _, guv, ginside = projection(camera, transform(slab, T))
    expected = _box_depth(camera, local, T)
    expected_mask = np.isfinite(expected) & (expected > 0.01)
    v, u = np.indices(expected.shape)
    k = camera.intrinsic
    rays = np.stack(
        ((u + 0.5 - k[0, 2]) / k[0, 0], (v + 0.5 - k[1, 2]) / k[1, 1], np.ones_like(u)),
        -1,
    )
    points_cam = rays * np.where(expected_mask, expected, 0)[..., None]
    points_world = transform(points_cam.reshape(-1, 3), camera.world_from_camera)
    points_local = (points_world - T[:3, 3]) @ T[:3, :3]
    ax = points_local[:, 0].reshape(expected.shape)
    regions = {
        "white_end": expected_mask & (ax <= -half[0] + pad_width),
        "red_end": expected_mask & (ax >= half[0] - pad_width),
        "grasp": expected_mask & (np.abs(ax - gx) <= gh),
    }
    observed = camera.depth
    difference = observed - expected
    finite = np.isfinite(observed)
    supported = expected_mask & finite & (np.abs(difference) <= tolerance)
    nearer = expected_mask & finite & (difference < -tolerance)
    farther = expected_mask & finite & (difference > tolerance)
    invalid = expected_mask & ~finite
    frame = {
        "white_end": bool(inside[SIGNS[:, 0] < 0].all()),
        "red_end": bool(inside[SIGNS[:, 0] > 0].all()),
        "grasp": bool(ginside.all()),
    }
    details = {}
    for name, mask in regions.items():
        n = int(mask.sum())
        n_ok = int((mask & supported).sum())
        details[name] = dict(
            in_frame=frame[name],
            pixels=n,
            supported=n_ok,
            nearer_pixels=int((mask & nearer).sum()),
            farther_pixels=int((mask & farther).sum()),
            invalid_pixels=int((mask & invalid).sum()),
            visible_fraction=n_ok / n if n else None,
            passed=bool(frame[name] and n > 0 and n_ok == n),
        )
    reasons = []
    for name, r in details.items():
        if not r["in_frame"]:
            reasons.append(name + "_out_of_frame")
        if r["pixels"] == 0:
            reasons.append(name + "_unresolved")
        if r["nearer_pixels"]:
            reasons.append(name + "_occluded")
        if r["farther_pixels"] or r["invalid_pixels"]:
            reasons.append(name + "_depth_unconfirmed")
    requested = all(r["passed"] for r in details.values())
    n = int(expected_mask.sum())
    n_ok = int(supported.sum())
    strict = bool(requested and inside.all() and n > 0 and n_ok == n)
    sensitivity = {
        str(t): bool(
            inside.all()
            and n > 0
            and all(r["pixels"] > 0 for r in details.values())
            and np.all(finite[expected_mask])
            and np.all(np.abs(difference[expected_mask]) <= t)
        )
        for t in (0.001, 0.0015, 0.002)
    }
    result = dict(
        requested_pass=bool(requested),
        strict_100_pass=strict,
        full_in_frame=bool(inside.all()),
        all_corners_in_front=bool((xyz[:, 2] > 0.01).all()),
        reasons=reasons,
        regions=details,
        expected_in_frame_pixels=n,
        depth_supported_pixels=n_ok,
        supported_fraction_in_frame=n_ok / n if n else None,
        nearer_pixels=int(nearer.sum()),
        farther_pixels=int(farther.sum()),
        invalid_pixels=int(invalid.sum()),
        grasp_world=target.tolist(),
        grasp_local=g_local.tolist(),
        grasp_half_width_m=gh,
        corners_camera=xyz.tolist(),
        corners_uv=uv.tolist(),
        grasp_corners_uv=guv.tolist(),
        strict_tolerance_sensitivity=sensitivity,
    )
    evidence = dict(
        expected_depth=expected,
        expected_mask=expected_mask,
        supported=supported,
        nearer=nearer,
        farther=farther,
        invalid=invalid,
        **regions,
    )
    return result, evidence
