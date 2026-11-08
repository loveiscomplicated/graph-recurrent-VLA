"""Evaluation-only physical geometry and verified simulator seed replay.

Canonical identities are (family, sign_x, sign_y, sign_z), never detector order.
No nearest-neighbour assignment is used to change an observed graph.
"""

from itertools import product
from pathlib import Path
import importlib.metadata
import json

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.episode_dataset import file_sha256


SIGNS = np.asarray(list(product((-1, 1), repeat=3)), dtype=np.float64)


def rotation_matrix(pose):
    """Convert a world pose's normalized wxyz quaternion to a rotation matrix."""
    quaternion = np.asarray(pose)[3:]
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def transform_points(local_points, pose):
    """Apply world pose [xyz, qwxyz] to local metre coordinates."""
    return np.asarray(local_points) @ rotation_matrix(pose).T + np.asarray(pose)[:3]


def physical_vertices(box_pose, hole_pose, half_length, hole_radius):
    """Return eight exterior corners and eight corners of the two aperture mouths.

    The simulator Hole origin is the tunnel midpoint, not either visible mouth.
    Exterior cube extent is independent of the randomized local hole offset.
    """
    outer = transform_points(SIGNS * half_length, box_pose)
    rim = transform_points(SIGNS * [half_length, hole_radius, hole_radius], hole_pose)
    return outer, rim


def rigid_registration(local_points, world_points):
    """Fit a proper rigid transform to known correspondences; reject collinearity.

    Three non-collinear labelled points suffice, including a planar triangle.
    An unlabelled cube still has discrete rotation ambiguities; this function
    makes no correspondence inference and does not resolve those ambiguities.
    """
    local_points = np.asarray(local_points, dtype=np.float64)
    world_points = np.asarray(world_points, dtype=np.float64)
    if local_points.shape != world_points.shape or local_points.ndim != 2 or local_points.shape[1] != 3:
        raise ValueError("Corresponding local/world points must both have shape [N,3]")
    if len(local_points) < 3 or not np.isfinite(local_points).all() or not np.isfinite(world_points).all():
        raise ValueError("Rigid fitting requires at least three finite corresponding points")
    centered_local = local_points - local_points.mean(0)
    centered_world = world_points - world_points.mean(0)
    if np.linalg.matrix_rank(centered_local, tol=1e-9) < 2:
        raise ValueError("Collinear or repeated physical corners do not determine a rigid pose")
    left, _, right = np.linalg.svd(centered_local.T @ centered_world)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(right.T @ left.T)
    rotation = right.T @ correction @ left.T
    translation = world_points.mean(0) - rotation @ local_points.mean(0)
    return rotation, translation


def verify_replay(state, targets, tcp_world, tolerance_m):
    """Accept static fixture replay only after matching every saved frame's GT.

    Matching Hole pose and dimensions alone cannot validate hidden geometry.
    Collection source must also establish that BOX is kinematic and is never
    teleported by its policy. The report retains this provenance limitation.
    """
    hole = np.asarray(state["hole_pose"]).reshape(7)
    hole_world = targets[:, 10:13].astype(np.float64) + tcp_world.astype(np.float64)
    position_error = float(np.max(np.linalg.norm(hole_world - hole[:3], axis=1)))
    size_error = float(np.max(np.abs(targets[:, 17:20] - state["peg_half_sizes"])))
    radius_error = float(np.max(np.abs(targets[:, 20] - state["hole_radius"])))
    saved_rotation = Rotation.from_quat(targets[:, [14, 15, 16, 13]])
    replay_rotation = Rotation.from_quat(hole[[4, 5, 6, 3]])
    rotation_error = float(np.max((saved_rotation.inv() * replay_rotation).magnitude()))
    if max(position_error, size_error, radius_error) > tolerance_m or rotation_error > 1e-5:
        raise ValueError(f"Seed replay does not match validation GT: position={position_error:g}m, "
                         f"size={size_error:g}m, radius={radius_error:g}m, rotation={rotation_error:g}rad")
    return dict(position_max_m=position_error, size_max_m=size_error,
                radius_max_m=radius_error, rotation_max_rad=rotation_error)


def replay_validation(config, geometry, identity, parent_manifest, output):
    """Reset only validation seeds and verify static BOX geometry against saved GT."""
    from envs.maniskill import get_gt_state, make_peg_insertion_env
    import mani_skill.envs.tasks.tabletop.peg_insertion_side as task_source
    import sapien.physx as physx
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes

    episodes = np.asarray([str(Path(value).parent) for value in geometry["sample_ids"]])
    expected = set(identity["splits"]["validation"])
    if set(episodes) != expected or expected & set(identity["splits"]["test"]):
        raise ValueError("Geometry must contain the complete original validation split only")
    parents = parent_manifest["parent_by_episode"]
    parent_sets = {split: {parents[ep] for ep in values} for split, values in identity["splits"].items()}
    if any(parent_sets[a] & parent_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Original parent splits overlap")
    manifest = json.loads(Path(config["collection_manifest"]).read_text())
    seeds = {row["path"]: row["seed"] for row in manifest["episodes"]}
    source_path = Path(task_source.__file__)
    evidence = dict(task_source_path=str(source_path), task_source_sha256=file_sha256(source_path),
                    packages={name: importlib.metadata.version(name) for name in ("mani_skill", "sapien", "torch", "numpy")},
                    episodes={}, mesh_checks=[])
    (output / "installed_peg_insertion_side.py").write_bytes(source_path.read_bytes())
    states = {}
    env = make_peg_insertion_env(device="cpu", obs_mode="state", control_mode="pd_ee_delta_pose")
    try:
        evidence["backend"] = {name: str(getattr(env.unwrapped.backend, name))
                               for name in ("sim_backend", "render_backend", "device")}
        for number, episode in enumerate(sorted(expected)):
            env.reset(seed=seeds[episode])
            raw = get_gt_state(env)
            state = {key: getattr(raw, key).cpu().numpy()[0]
                     for key in ("box_pose", "hole_pose", "peg_half_sizes", "hole_radius")}
            indices = np.flatnonzero(episodes == episode)
            checks = verify_replay(state, geometry["target"][indices], geometry["tcp_world"][indices],
                                   float(config.get("replay_tolerance_m", 2e-6)))
            states[episode] = {key: np.asarray(value).tolist() for key, value in state.items()}
            evidence["episodes"][episode] = dict(seed=seeds[episode], frames=len(indices), **checks)
            if number < 3:
                meshes = [mesh for component in env.unwrapped.box._objs[0].components
                          if isinstance(component, physx.PhysxRigidBaseComponent)
                          for mesh in get_component_meshes(component)]
                mesh_world = transform_points(np.concatenate([mesh.vertices for mesh in meshes]), state["box_pose"])
                outer, rim = physical_vertices(state["box_pose"], state["hole_pose"],
                                               state["peg_half_sizes"][0], state["hole_radius"])
                distances = np.linalg.norm(outer[:, None] - mesh_world[None], axis=2).min(1)
                # The four overlapping slabs do not put a triangulation vertex
                # at the aperture's concave corner. Recover their intersection
                # from slab bounds instead of mistaking mesh tessellation IDs
                # for physical rim identities.
                bounds = np.asarray([mesh.bounds for mesh in meshes])
                positive_y, negative_y = bounds[0, 0, 1], bounds[1, 1, 1]
                positive_z, negative_z = bounds[2, 0, 2], bounds[3, 1, 2]
                aperture_local = np.column_stack((SIGNS[:, 0] * state["peg_half_sizes"][0],
                    np.where(SIGNS[:, 1] > 0, positive_y, negative_y),
                    np.where(SIGNS[:, 2] > 0, positive_z, negative_z)))
                aperture_world = transform_points(aperture_local, state["box_pose"])
                rim_error = float(np.linalg.norm(aperture_world - rim, axis=1).max())
                if max(distances.max(), rim_error) > 2e-6:
                    raise ValueError("Canonical outer/rim vertices disagree with installed collision geometry")
                evidence["mesh_checks"].append(dict(episode=episode, shapes=len(meshes),
                    outer_vertex_max_error_m=float(distances.max()), aperture_intersection_max_error_m=rim_error))
            if number % 20 == 0:
                print(f"Verified validation seed replay: {number + 1}/{len(expected)}", flush=True)
    finally:
        env.close()
    return states, evidence


def geometric_oracles(states, seed=0):
    """Measure labelled GT fitting, hidden-offset ambiguity and controlled noise.

    Each row is one validation parent, not one independent video frame. These
    are GT-geometry diagnostics, not Student inference or detection correction.
    """
    random = np.random.default_rng(seed)
    rows = []
    subsets = {"triangle": [0, 1, 2], "one_face": [0, 1, 2, 3], "full": list(range(8))}
    for episode, state in states.items():
        box = np.asarray(state["box_pose"])
        hole = np.asarray(state["hole_pose"])
        length = state["peg_half_sizes"][0]
        radius = state["hole_radius"]
        outer, rim = physical_vertices(box, hole, length, radius)
        offset = rotation_matrix(box).T @ (hole[:3] - box[:3])
        for family, local, points in (("outer", SIGNS * length, outer),
                                      ("rim", SIGNS * [length, radius, radius], rim)):
            for subset, indices in subsets.items():
                for sigma in (0, .0005, .001):
                    for repeat in range(1 if sigma == 0 else 20):
                        observed = points[indices] + random.normal(0, sigma, (len(indices), 3))
                        rotation, translation = rigid_registration(local[indices], observed)
                        predicted_hole = translation + rotation @ offset if family == "outer" else translation
                        angle = Rotation.from_matrix(rotation.T @ rotation_matrix(hole)).magnitude()
                        rows.append(dict(episode=episode, family=family, subset=subset, sigma_mm=sigma * 1000,
                                         repeat=repeat, hole_error_mm=float(np.linalg.norm(predicted_hole - hole[:3]) * 1000),
                                         rotation_error_deg=float(np.rad2deg(angle)),
                                         box_center_to_hole_mm=float(np.linalg.norm(offset) * 1000)))
    return rows
