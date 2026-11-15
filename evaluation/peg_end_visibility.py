"""Audit lost Peg identities and compare an observed-end silhouette gate."""

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from envs.maniskill import make_peg_insertion_env
from evaluation.trajectory_rim_recovery import load_frame
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import score, visible_reference
from visual.corner_detection.peg_prism import prism_corners


class ObservedEndDetector(TorchCornerDetector):
    """Allow silhouette correction when the current fit contains an end plane."""

    def peg_vertices(self, data):
        return prism_corners(
            data.points.numpy(),
            data.planes_cpu,
            silhouette=(data.support_cpu, self.depth.numpy(), self.k_cpu),
            require_end_face=True,
        )


class TracedSilhouetteDetector(TorchCornerDetector):
    """Capture actual fitted bounds before corner filtering and assignment."""

    preserve_transverse_extent = False
    preserve_lower_silhouette = False

    def peg_vertices(self, data):
        self.bounds = None
        return prism_corners(
            data.points.numpy(),
            data.planes_cpu,
            silhouette=(data.support_cpu, self.depth.numpy(), self.k_cpu),
            preserve_transverse_extent=self.preserve_transverse_extent,
            trace_bounds=self.record_bounds,
            preserve_lower_silhouette=self.preserve_lower_silhouette,
        )

    def record_bounds(self, basis, before, after, anchored):
        self.bounds = basis, before, after, anchored


class PreserveExtentDetector(TracedSilhouetteDetector):
    preserve_transverse_extent = True


class PreserveLowerDetector(TracedSilhouetteDetector):
    preserve_lower_silhouette = True


def bound_errors(bounds, sensor, gt):
    """Compare each fitted support plane against the GT support in the same direction."""
    basis, before, after, anchored = bounds
    ext = sensor["extrinsic_cv"]
    pose = gt["peg_pose"]
    rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    alignment = basis @ ext[:, :3] @ rotation
    center = basis @ (ext[:, :3] @ pose[:3] + ext[:, 3])
    half_span = np.abs(alignment) @ gt["peg_half_sizes"]
    truth = np.stack((center - half_span, center + half_span))
    rows = []
    for side, axis in zip(*np.nonzero(before != after), strict=True):
        local_axis = np.argmax(np.abs(alignment[axis]))
        sign = int((2 * side - 1) * np.sign(alignment[axis, local_axis]))
        rows.append(
            {
                "axis": "xyz"[local_axis],
                "camera_down": float((2 * side - 1) * basis[axis, 1]),
                "side": sign,
                "inward_shift_mm": (after[side, axis] - before[side, axis])
                * (1 - 2 * side)
                * 1000,
                "before_error_mm": abs(before[side, axis] - truth[side, axis]) * 1000,
                "after_error_mm": abs(after[side, axis] - truth[side, axis]) * 1000,
                "opposite_face_anchored": bool(anchored[1 - side, axis]),
            }
        )
    return rows


def initialize():
    """Create separate CPU workspaces for the paired detector variants."""
    global detectors
    torch.set_num_threads(1)
    detectors = {
        name: cls(
            "cpu",
            recover_small_geometry=True,
            recover_hole_rim=True,
            recover_fragmented_rim=True,
        )
        for name, cls in (
            ("base", TorchCornerDetector),
            ("silhouette", TracedSilhouetteDetector),
            ("preserve_extent", PreserveExtentDetector),
            ("preserve_lower", PreserveLowerDetector),
            ("observed_end", ObservedEndDetector),
        )
    }


def end_occlusion(sensor, pose, half, rotation, robot_ids):
    """Count sampled end-face rays with a robot surface in front of the GT cap."""
    grid = np.linspace(-0.9, 0.9, 7)
    y, z = np.meshgrid(grid * half[1], grid * half[2])
    ext, intrinsic = sensor["extrinsic_cv"], sensor["intrinsic_cv"]
    depth = sensor["depth"].squeeze() * 0.001
    segmentation = sensor["segmentation"].squeeze()
    blocked = []
    for sign in (-1, 1):
        local = np.column_stack((np.full(y.size, sign * half[0]), y.ravel(), z.ravel()))
        camera = (local @ rotation.T + pose[:3]) @ ext[:, :3].T + ext[:, 3]
        uv = camera @ intrinsic.T
        u, v = np.rint(uv[:, :2] / uv[:, 2:]).astype(int).T
        inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        observed = depth[v[inside], u[inside]]
        actor = segmentation[v[inside], u[inside]]
        occluded = np.isin(actor, robot_ids) & (observed > 0)
        occluded &= observed < camera[inside, 2] - 0.002
        blocked.append(int(occluded.sum()))
    return blocked


def audit_frame(row, *, root, robot_ids, expected_actors):
    """Compare one-to-one Peg matches and audit the nearest surviving candidate."""
    sensor, gt, grasp = load_frame(root, row)
    ids = {int(key): value for key, value in row["actor_ids"].items()}
    if ids != expected_actors:
        raise ValueError("Benchmark actor IDs differ from the live robot ID check")
    references = [r for r in visible_reference(gt, sensor, ids) if r["object_id"] == 0]
    args = [sensor[key] for key in ("rgb", "depth", "intrinsic_cv", "extrinsic_cv")]
    predictions, matches, metrics = {}, {}, {}
    for name, detector in detectors.items():
        result = detector(*args)
        points = [p for p in result.as_dict()["corners"] if p["object_id"] == 0]
        matched = score(points, references, references)[0]
        predictions[name] = np.asarray([p["xyz"] for p in points]).reshape(-1, 3)
        matches[name] = {m["reference"]: m["prediction"] for m in matched["matches"]}
        metrics[name] = {
            "matched": matched["matched_3mm"],
            "reference": len(references),
            "unmatched": len(points) - matched["matched_3mm"],
        }
    bounds = detectors["silhouette"].bounds
    boundaries = bound_errors(bounds, sensor, gt) if bounds is not None else []
    losses = []
    if grasp:
        pose = gt["peg_pose"]
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        blocked = end_occlusion(sensor, pose, gt["peg_half_sizes"], rotation, robot_ids)
        for index in sorted(matches["base"].keys() - matches["silhouette"].keys()):
            ref = references[index]
            sign = 1 if ref["corner_id"] >= 4 else -1
            before = (
                predictions["base"][matches["base"][index]] - ref["xyz"]
            ) @ rotation
            offsets = (predictions["silhouette"] - ref["xyz"]) @ rotation
            after = (
                offsets[np.linalg.norm(offsets, axis=1).argmin()]
                if len(offsets)
                else None
            )
            losses.append(
                {
                    "corner_id": ref["corner_id"],
                    "robot_blocked_samples": blocked[int(sign > 0)],
                    "before_local_mm": before * 1000,
                    "after_local_mm": None if after is None else after * 1000,
                    "inward_delta_mm": None
                    if after is None
                    else -sign * (after[0] - before[0]) * 1000,
                }
            )
    return {
        "seed": row["seed"],
        "mode": row["mode"],
        "frame": row["frame"],
        "grasp": grasp,
        "metrics": metrics,
        "lost": losses,
        "boundaries": boundaries,
    }


def evaluate(config):
    """Replay the frozen trajectory cohort and persist the identity audit."""
    root = Path(config["dataset"])
    source = root / "trajectory.json"
    rows = json.loads(source.read_text())["rows"]
    env = make_peg_insertion_env(device="cpu", obs_mode="rgb+depth+segmentation")
    try:
        env.reset(seed=rows[0]["seed"])
        task = env.unwrapped
        robot_ids = [int(link.per_scene_id[0]) for link in task.agent.robot.links]
        actors = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
    finally:
        env.close()
    audit = partial(audit_frame, root=root, robot_ids=robot_ids, expected_actors=actors)
    with ExperimentLogger(
        team="graph",
        model="peg_end_visibility",
        config=config,
        seed=20261114,
        job_type="evaluation",
    ) as logger:
        with ProcessPoolExecutor(
            max_workers=config["workers"], initializer=initialize
        ) as pool:
            results = list(pool.map(audit, rows, chunksize=4))
        totals = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for row in results:
            group = "grasp" if row["grasp"] else "non_grasp"
            for variant, metrics in row["metrics"].items():
                for name, count in metrics.items():
                    totals[group][variant][name] += count
        path = Path(config["output_root"]) / logger.run_id / "results.json"
        path.parent.mkdir(parents=True, exist_ok=False)
        write_json(
            path,
            {
                "rows": results,
                "totals": totals,
                "robot_ids": robot_ids,
                "source_sha256": file_sha256(source),
            },
        )
        logger.link_output(path)
        print(path, json.dumps(totals), flush=True)
