"""Conservative convex-hull and interval checks for the fixed RoArm bench."""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import trimesh
from sapien.wrapper.pinocchio_model import PinocchioModel

from .assets import (
    ASSET_DIR,
    JOINT_NAMES,
    PHYSICAL_LINK_NAMES,
    URDF_PATH,
    load_collision_exclusions,
    verify_asset_hashes,
)
from .kinematics import RoArmKinematics
from .scene import validate_obstacles


class MotionRejected(RuntimeError):
    """The requested motion cannot be accepted under the current guard settings."""


@dataclass(frozen=True)
class SafetySettings:
    """Selected simulation margins and limits, not hardware specifications."""

    clearance_m: float = 0.005
    tracking_error_rad: float = 0.01
    command_speed_rad_s: float = 0.25
    command_acceleration_rad_s2: float = 0.5
    measured_speed_limit_rad_s: float = 0.4
    contact_limit_n: float = 0.1
    max_edge_depth: int = 12

    def __post_init__(self):
        for name in (
            "clearance_m",
            "tracking_error_rad",
            "command_speed_rad_s",
            "command_acceleration_rad_s2",
            "measured_speed_limit_rad_s",
            "contact_limit_n",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.max_edge_depth <= 20:
            raise ValueError("max_edge_depth must be between 1 and 20")


def _load_collision_meshes(root):
    """Load audited convex hulls and their largest local vertex radius."""
    meshes = {}
    max_vertex_radius = 0.0
    for link in root.findall("link"):
        name = link.get("name")
        if name not in PHYSICAL_LINK_NAMES:
            continue
        collision = link.find("collision")
        origin = collision.find("origin")
        # Only the audited zero-origin, unit-scale geometry is supported.
        if origin is not None:
            translation = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rotation = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            if translation.any() or rotation.any():
                raise ValueError(
                    "Collision origins changed; re-audit the collision model"
                )
        mesh_node = collision.find("geometry/mesh")
        if mesh_node.get("scale", "1 1 1") != "1 1 1":
            raise ValueError("Mesh scale changed; re-audit the collision model")
        hull = trimesh.load(
            ASSET_DIR / mesh_node.get("filename"), force="mesh"
        ).convex_hull
        vertices = np.asarray(hull.vertices)
        normals = np.unique(np.round(hull.face_normals, 10), axis=0)
        meshes[name] = (vertices, normals)
        max_vertex_radius = max(
            max_vertex_radius, float(np.linalg.norm(vertices, axis=1).max())
        )
    return meshes, max_vertex_radius


def _build_motion_bounds(root, max_vertex_radius):
    """Bound each link's vertex displacement per ancestor joint radian."""
    joints = root.findall("joint")
    # Triangle inequality: sum all joint translations plus mesh radius bounds
    # the distance from any ancestor revolute joint to any mesh vertex.
    radius = max_vertex_radius + sum(
        np.linalg.norm(np.fromstring(joint.find("origin").get("xyz"), sep=" "))
        for joint in joints
    )
    child_joints = {joint.find("child").get("link"): joint for joint in joints}
    motion_bounds = {}
    for name in PHYSICAL_LINK_NAMES:
        bounds = np.zeros(4)
        current_link = name
        while current_link in child_joints:
            joint = child_joints[current_link]
            joint_name = joint.get("name")
            if joint_name in JOINT_NAMES:
                bounds[JOINT_NAMES.index(joint_name)] = radius
            current_link = joint.find("parent").get("link")
        motion_bounds[name] = bounds
    return motion_bounds


def _separation_gap(first_vertices, second_vertices, axes):
    """Return the largest signed gap along the candidate separating axes."""
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    first_projection = first_vertices @ axes.T
    second_projection = second_vertices @ axes.T
    gaps = np.maximum(
        second_projection.min(axis=0) - first_projection.max(axis=0),
        first_projection.min(axis=0) - second_projection.max(axis=0),
    )
    return gaps.max()


class RoArmCollisionModel:
    """Reject any pose or interval without a proven separating projection.

    Hull face normals and world axes provide a conservative test: omitting
    edge-cross-edge axes can reject some free configurations.
    """

    link_names = PHYSICAL_LINK_NAMES

    def __init__(self, settings=None, obstacles=()):
        self.settings = settings or SafetySettings()
        self.kinematics = RoArmKinematics()
        verify_asset_hashes()
        root = ET.parse(URDF_PATH).getroot()
        self.model = PinocchioModel(URDF_PATH.read_text(), [0, 0, -9.81])
        self.model.set_joint_order(JOINT_NAMES)
        self.model.set_link_order(self.link_names)
        self.meshes, max_vertex_radius = _load_collision_meshes(root)
        self.motion_bounds = _build_motion_bounds(root, max_vertex_radius)
        self.obstacles = validate_obstacles(obstacles)
        excluded = load_collision_exclusions()
        self.pairs = [
            (first, second)
            for first, second in itertools.combinations(self.link_names, 2)
            if frozenset((first, second)) not in excluded
        ]

    @staticmethod
    def validate_qpos(value):
        """Return one finite four-joint configuration as a NumPy array."""
        qpos = np.asarray(value, dtype=float)
        if qpos.shape != (4,) or not np.isfinite(qpos).all():
            raise MotionRejected("Expected four finite joint angles")
        return qpos

    def geometry(self, qpos):
        """Return base-frame convex hull vertices/normals at one configuration."""
        self.model.compute_forward_kinematics(self.validate_qpos(qpos))
        geometry = {}
        for index, name in enumerate(self.link_names):
            transform = self.model.get_link_pose(index).to_transformation_matrix()
            vertices, normals = self.meshes[name]
            geometry[name] = (
                vertices @ transform[:3, :3].T + transform[:3, 3],
                normals @ transform[:3, :3].T,
            )
        return geometry

    def rejection(self, qpos, half_span=None):
        """Return a reason, or None if the whole joint-space box is separated.

        half_span encloses a path interval around qpos. Vertex movement is bounded
        by sum(radius_j * angle_j), including tracking error on each joint.
        """
        qpos = self.validate_qpos(qpos)
        span = np.zeros(4) if half_span is None else np.asarray(half_span, dtype=float)
        if span.shape != (4,) or not np.isfinite(span).all() or (span < 0).any():
            raise ValueError("half_span must contain four finite nonnegative angles")
        joint_margin = span + self.settings.tracking_error_rad
        limits = self.kinematics.limits
        if (qpos - joint_margin < limits[:, 0]).any() or (
            qpos + joint_margin > limits[:, 1]
        ).any():
            return "joint limits including tracking margin"
        geometry = self.geometry(qpos)
        clearance = self.settings.clearance_m + 1e-6  # float32 Pose allowance

        # The tabletop is conservatively an infinite plane at base-frame z=0.
        for name in self.link_names[1:]:
            movement = float(self.motion_bounds[name] @ joint_margin)
            if geometry[name][0][:, 2].min() <= clearance + movement:
                return f"{name}/table clearance"

        for index, obstacle in enumerate(self.obstacles):
            corner_signs = np.array(list(itertools.product((-1, 1), repeat=3)))
            box_vertices = corner_signs * obstacle["half_size"] + obstacle["center"]
            for name in self.link_names:
                vertices, normals = geometry[name]
                axes = np.vstack([np.eye(3), normals])
                gap = _separation_gap(vertices, box_vertices, axes)
                movement = self.motion_bounds[name] @ joint_margin
                if gap <= clearance + movement:
                    return f"{name}/guard_obstacle_{index} clearance"

        for first, second in self.pairs:
            first_vertices, first_normals = geometry[first]
            second_vertices, second_normals = geometry[second]
            axes = np.vstack([np.eye(3), first_normals, second_normals])
            gap = _separation_gap(first_vertices, second_vertices, axes)
            movement = float(
                (self.motion_bounds[first] + self.motion_bounds[second]) @ joint_margin
            )
            if gap <= clearance + movement:
                return f"{first}/{second} clearance"
        return None

    def edge_valid(self, start, goal, depth=0):
        """Certify a linear joint path by recursively bounding entire intervals."""
        start = self.validate_qpos(start)
        goal = self.validate_qpos(goal)
        midpoint = (start + goal) / 2
        if self.rejection(midpoint, np.abs(goal - start) / 2) is None:
            return True
        if (
            depth >= self.settings.max_edge_depth
            or self.rejection(midpoint) is not None
        ):
            return False
        return self.edge_valid(start, midpoint, depth + 1) and self.edge_valid(
            midpoint, goal, depth + 1
        )

    def export_hulls(self, qpos, path):
        """Export the actual collision hulls for inspection beside the render."""
        scene = trimesh.Scene()
        for name, (vertices, _) in self.geometry(qpos).items():
            scene.add_geometry(trimesh.convex.convex_hull(vertices), node_name=name)
        scene.export(str(path))
