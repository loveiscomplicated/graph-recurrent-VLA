"""Observed vertex-family annotations for a separate development dataset.

Detector labels use retained proposal metadata only. Oracle labels require a
unique GT corner cell, calibrated projection agreement, and independent RGB-D
support for the physical face's occupied/empty quadrants. Failed evidence stays
UNKNOWN; no nearest-reference fallback or coordinate correction is performed.
"""

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


UNKNOWN = -1
EXTERIOR = 0
RIM = 1
SIGNS = np.asarray(list(product((-1, 1), repeat=3)), dtype=np.float64)


@dataclass(frozen=True)
class CertificationSettings:
    """Predeclared bounds; they are not selected on development validation error."""

    coordinate_tolerance_m: float = 0.003
    projection_tolerance_px: float = 2.0
    face_depth_tolerance_m: float = 0.0015
    probe_offsets_m: tuple = (0.004, 0.006)

    def __post_init__(self):
        values = (self.coordinate_tolerance_m, self.projection_tolerance_px,
                  self.face_depth_tolerance_m, *self.probe_offsets_m)
        if not values or any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Certification tolerances and probe offsets must be finite and positive")


def detector_families(object_ids, metadata):
    """Map the detector's contour claim to family IDs without consulting GT."""
    if len(object_ids) != len(metadata):
        raise ValueError("One retained detector metadata record is required per vertex")
    labels = np.full(len(object_ids), UNKNOWN, dtype=np.int64)
    for index, (object_id, record) in enumerate(zip(object_ids, metadata)):
        if object_id != 1:
            continue
        hole_flag = record.get("contour_hole")
        if isinstance(hole_flag, (bool, np.bool_)):
            labels[index] = RIM if hole_flag else EXTERIOR
    return labels


def pose_rotation(pose):
    """Read a simulator world pose with quaternion order qw,qx,qy,qz."""
    quaternion = np.asarray(pose).reshape(7)[3:]
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def project_world(points, intrinsic, extrinsic):
    """Project metre world coordinates into the recorded CV camera."""
    camera = np.asarray(points) @ extrinsic[:, :3].T + extrinsic[:, 3]
    image = camera @ intrinsic.T
    return image[:, :2] / image[:, 2:3], camera


def physical_references(ground_truth):
    """Return labelled cube corners and the two physical aperture mouths."""
    box_pose = np.asarray(ground_truth["box_pose"]).reshape(7)
    hole_pose = np.asarray(ground_truth["hole_pose"]).reshape(7)
    length = np.asarray(ground_truth["peg_half_sizes"]).reshape(3)[0]
    radius = float(np.asarray(ground_truth["hole_radius"]).reshape(-1)[0])
    references = []
    for family, pose, scales in ((EXTERIOR, box_pose, [length] * 3),
                                 (RIM, hole_pose, [length, radius, radius])):
        rotation = pose_rotation(pose)
        for identity, sign in enumerate(SIGNS):
            local = sign * scales
            references.append(dict(family=family, physical_id=family * 8 + identity,
                                   local=local, rotation=rotation, origin=pose[:3],
                                   world=local @ rotation.T + pose[:3]))
    return references


def box_material_mask(rgb):
    """Identify the task's yellow fixture material using RGB only."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 0] >= 14) & (hsv[..., 0] <= 30)
            & (hsv[..., 1] >= 65) & (hsv[..., 1] <= 195) & (hsv[..., 2] >= 50))


def face_topology_supported(reference, observation, ground_truth, settings, material=None):
    """Verify GT physical corner topology using the raw image, not detector flags.

    At each of two distances, probe all four local yz quadrants on the mouth
    plane. An exterior corner has one occupied quadrant; a rim corner has three.
    Recorded material and raw depth must agree with that GT occupancy pattern.
    A back mouth occluded by the front face is rejected by the plane-depth test.
    """
    intrinsic = observation["intrinsic_cv"]
    extrinsic = observation["extrinsic_cv"]
    depth = np.asarray(observation["depth"]).squeeze()
    if material is None:
        material = box_material_mask(observation["rgb"])
    box_pose = np.asarray(ground_truth["box_pose"]).reshape(7)
    hole_pose = np.asarray(ground_truth["hole_pose"]).reshape(7)
    length = np.asarray(ground_truth["peg_half_sizes"]).reshape(3)[0]
    radius = float(np.asarray(ground_truth["hole_radius"]).reshape(-1)[0])
    box_rotation = pose_rotation(box_pose)
    hole_rotation = pose_rotation(hole_pose)
    camera_normal = extrinsic[:, :3] @ reference["rotation"][:, 0]
    corner_camera = extrinsic[:, :3] @ reference["world"] + extrinsic[:, 3]
    plane_offset = -camera_normal @ corner_camera
    quadrants = np.asarray(list(product((-1, 1), repeat=2)))
    inverse_intrinsic = np.linalg.inv(intrinsic)
    for offset in settings.probe_offsets_m:
        local_probes = np.repeat(reference["local"][None], 4, axis=0)
        local_probes[:, 1:] += quadrants * offset
        world = local_probes @ reference["rotation"].T + reference["origin"]
        box_local = (world - box_pose[:3]) @ box_rotation
        hole_local = (world - hole_pose[:3]) @ hole_rotation
        within_cube = (np.abs(box_local[:, 1:]) < length).all(1)
        within_aperture = (np.abs(hole_local[:, 1:]) < radius).all(1)
        expected_occupied = within_cube & ~within_aperture
        expected_count = 1 if reference["family"] == EXTERIOR else 3
        if expected_occupied.sum() != expected_count:
            return False
        projected, camera = project_world(world, intrinsic, extrinsic)
        pixels = np.rint(projected).astype(int)
        if (camera[:, 2] <= 0).any() or (pixels < 0).any():
            return False
        if (pixels[:, 0] >= depth.shape[1]).any() or (pixels[:, 1] >= depth.shape[0]).any():
            return False
        raw = depth[pixels[:, 1], pixels[:, 0]]
        if (raw <= 0).any() or (raw >= 4000).any():
            return False
        rays = np.column_stack((pixels, np.ones(4))) @ inverse_intrinsic.T
        denominator = rays @ camera_normal
        if (np.abs(denominator) < .08).any():
            return False
        plane_depth = -plane_offset / denominator
        depth_agrees = np.abs(raw * .001 - plane_depth) <= settings.face_depth_tolerance_m
        measured_occupied = material[pixels[:, 1], pixels[:, 0]] & depth_agrees
        if not np.array_equal(measured_occupied, expected_occupied):
            return False
    return True


def oracle_families(xyz, uv, object_ids, observation, ground_truth, settings=None):
    """Certify only observed nodes with unique geometry and independent image evidence.

    The returned ID denotes a GT physical corner only when all gates pass. This
    is an operational finite-tolerance certificate, not a proof against arbitrary
    sensor/model error. Uncertified nodes are retained with unchanged XYZ.
    """
    settings = settings or CertificationSettings()
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)
    object_ids = np.asarray(object_ids)
    if xyz.shape != (len(object_ids), 3) or uv.shape != (len(object_ids), 2):
        raise ValueError("Observed XYZ, UV and object IDs must have matching [N,3], [N,2], [N] shapes")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Observed XYZ and UV must contain finite values")
    references = physical_references(ground_truth)
    family = np.full(len(xyz), UNKNOWN, dtype=np.int64)
    physical_id = np.full(len(xyz), UNKNOWN, dtype=np.int64)
    reasons = []
    material = box_material_mask(observation["rgb"])
    for index, (point, pixel, object_id) in enumerate(zip(xyz, uv, object_ids)):
        if object_id != 1:
            reasons.append("not_target")
            continue
        admissible = []
        for reference in references:
            local = (point - reference["origin"]) @ reference["rotation"]
            if np.max(np.abs(local - reference["local"])) <= settings.coordinate_tolerance_m:
                admissible.append(reference)
        if len(admissible) != 1:
            reasons.append("no_unique_physical_corner_cell")
            continue
        reference = admissible[0]
        projected, camera = project_world(reference["world"][None], observation["intrinsic_cv"], observation["extrinsic_cv"])
        if camera[0, 2] <= 0 or np.linalg.norm(projected[0] - pixel) > settings.projection_tolerance_px:
            reasons.append("projection_disagrees")
            continue
        if not face_topology_supported(reference, observation, ground_truth, settings, material):
            reasons.append("raw_face_topology_unverified")
            continue
        family[index] = reference["family"]
        physical_id[index] = reference["physical_id"]
        reasons.append("geometry_projection_and_raw_topology")
    return family, physical_id, reasons
