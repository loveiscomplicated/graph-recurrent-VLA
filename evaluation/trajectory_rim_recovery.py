"""Evaluate causal rim reuse on the frozen trajectory benchmark."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import corners, score, visible_reference
from visual.corner_detection.peg_prism import PegSizeMemory, prism_corners
from visual.corner_detection.rim_consensus import RimConsensus
from visual.corner_detection.rim_memory import StaticRimMemory


class PegSizeDetector(TorchCornerDetector):
    """Compare size-conditioned vertices without changing the frame-only path."""

    def peg_vertices(self, data):
        return prism_corners(
            data.points.numpy(), data.planes_cpu, size_memory=self.size_memory
        )


class PegSilhouetteDetector(TorchCornerDetector):
    """Refine unobserved prism bounds with current silhouette edges."""

    def peg_vertices(self, data):
        silhouette = (data.support_cpu, self.depth.numpy(), self.k_cpu)
        return prism_corners(
            data.points.numpy(), data.planes_cpu, silhouette=silhouette
        )


class LongSpanDetector(TorchCornerDetector):
    """Use the better-resolved span as the square side hypothesis."""

    _opening_square_span = "long"


class ShortSpanDetector(TorchCornerDetector):
    """Use the shorter span as the square side hypothesis."""

    _opening_square_span = "short"


def initialize():
    """Instantiate independent CPU detector workspaces in each worker."""
    global primary, alternatives, peg_detector, silhouette_detector
    torch.set_num_threads(1)
    options = {
        "device": "cpu",
        "recover_hole_rim": True,
        "recover_small_geometry": True,
        "recover_fragmented_rim": True,
    }
    primary = TorchCornerDetector(**options)
    peg_detector = PegSizeDetector(**options)
    silhouette_detector = PegSilhouetteDetector(**options)
    alternatives = [LongSpanDetector(**options), ShortSpanDetector(**options)]


def load_frame(root, row, *, require_grasp=True):
    """Read checksum-verified image inputs and evaluation GT."""
    path = root / row["path"]
    if file_sha256(path) != row["sha256"]:
        raise ValueError(f"Benchmark frame checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as frame:
        sensor = {
            name: frame[f"observation/sensor_data/base_camera/{name}"][0]
            for name in ("rgb", "depth", "segmentation")
        }
        sensor.update(
            {
                name: frame[f"observation/sensor_param/base_camera/{name}"][0]
                for name in ("intrinsic_cv", "extrinsic_cv")
            }
        )
        gt = {
            name: frame[f"ground_truth/{name}"][0]
            for name in (
                "peg_pose",
                "box_pose",
                "hole_pose",
                "peg_half_sizes",
                "hole_radius",
            )
        }
        grasp = bool(frame["collection/grasp"]) if require_grasp else None
    return sensor, gt, grasp


def observed_squares(detector, extrinsic):
    """Export complete verified rim fits before point-level merging."""
    indices = dict.fromkeys(
        row["block"] for row in detector.rows if row["meta"]["method"] == "box_opening"
    )
    observations = []
    for index in indices:
        points, _, _, valid, _ = detector.blocks[index]
        if len(points) == 4 and bool(valid.all()):
            observations.append((points.numpy() - extrinsic[:, 3]) @ extrinsic[:, :3])
    return observations


def depth_ranked_rims(
    primary_predictions, recovered, result, depth_mm, intrinsic, extrinsic
):
    """Prioritize supplemental rims near measured Box-to-recess boundaries."""
    depth = depth_mm.squeeze() * 0.001
    mask = result.masks[1].numpy()
    edges = []
    for axis in (0, 1):
        face = np.moveaxis(mask, axis, 0)
        values = np.moveaxis(depth, axis, 0)
        transition = (face[:-1] & (values[1:] > values[:-1] + 0.003)) | (
            face[1:] & (values[:-1] > values[1:] + 0.003)
        )
        row, col = np.nonzero(transition)
        uv = np.column_stack((col, row + 0.5) if axis == 0 else (row + 0.5, col))
        edges.append(uv)
    boundary = np.concatenate(edges)
    if not len(boundary):
        return recovered

    def distance(point):
        camera = extrinsic[:, :3] @ point["xyz"] + extrinsic[:, 3]
        uv = camera[:2] / camera[2] * np.diag(intrinsic)[:2] + intrinsic[:2, 2]
        return np.linalg.norm(boundary - uv, axis=1).min()

    kept = []
    for point in sorted(recovered[len(primary_predictions) :], key=distance):
        if all(
            np.linalg.norm(np.subtract(point["xyz"], q["xyz"])) >= 0.003 for q in kept
        ):
            kept.append(point)
    return [*primary_predictions, *kept]


def score_rollout(job):
    """Process frames chronologically with empty memory at the episode boundary."""
    root, frames = job
    memory = StaticRimMemory()
    consensus = RimConsensus()
    peg_detector.size_memory = PegSizeMemory()
    episodes = []
    for row in sorted(frames, key=lambda item: item["frame"]):
        sensor, gt, grasp = load_frame(root, row)
        args = tuple(
            sensor[key] for key in ("rgb", "depth", "intrinsic_cv", "extrinsic_cv")
        )
        result = primary(*args)
        squares = observed_squares(primary, args[-1])
        supplemental = []
        for detector in alternatives:
            supplemental.extend(
                p
                for p in detector(*args).as_dict()["corners"]
                if p["method"] == "box_opening"
            )
            squares.extend(observed_squares(detector, args[-1]))
        recovered = memory.update(result, *args[1:], supplemental)
        fitted = [
            {"xyz": point.tolist(), "object_id": 1, "method": "box_opening"}
            for point in consensus.update(squares)
        ]
        fitted_predictions = StaticRimMemory().update(result, *args[1:], fitted)
        primary_predictions = result.as_dict()["corners"]
        peg_predictions = peg_detector(*args).as_dict()["corners"]
        peg_size = list(primary_predictions)
        peg_points = [np.asarray(p["xyz"]) for p in peg_size if p["object_id"] == 0]
        for point in peg_predictions:
            if point["method"] != "peg_prism":
                continue
            xyz = np.asarray(point["xyz"])
            if all(np.linalg.norm(xyz - other) >= 0.002 for other in peg_points):
                peg_size.append(point)
                peg_points.append(xyz)
        silhouette_predictions = [
            p
            for p in silhouette_detector(*args).as_dict()["corners"]
            if p["object_id"] == 0
        ] + [p for p in recovered if p["object_id"] == 1]
        # Keep the current detector's predictions and collapse only supplemental points.
        ranked = list(recovered[: len(primary_predictions)])
        center_fit = np.asarray([p["xyz"] for p in fitted]).reshape(-1, 3)
        if len(center_fit):
            extras = recovered[len(primary_predictions) :]
            extras.sort(
                key=lambda p: np.linalg.norm(center_fit - p["xyz"], axis=1).min()
            )
            kept = []
            for point in extras:
                if all(
                    np.linalg.norm(np.subtract(point["xyz"], q["xyz"])) >= 0.003
                    for q in kept
                ):
                    kept.append(point)
            ranked.extend(kept)
        else:
            ranked = recovered
        depth_ranked = depth_ranked_rims(
            primary_predictions, recovered, result, *args[1:]
        )
        ids = {int(key): value for key, value in row["actor_ids"].items()}
        references = visible_reference(gt, sensor, ids)
        physical = corners(gt)
        scores = {}
        for name, predictions in (
            ("frame_only", result.as_dict()["corners"]),
            ("temporal", recovered),
            ("consensus", fitted_predictions),
            ("peg_size", peg_size),
            ("ranked_rim", ranked),
            ("peg_silhouette", silhouette_predictions),
            ("depth_ranked_rim", depth_ranked),
        ):
            peg, box = score(predictions, references, physical)
            families = box["by_kind"]
            scores[name] = {
                "rim_matched": families.get("hole_rim", {}).get("matched", 0),
                "rim_reference": families.get("hole_rim", {}).get("visible", 0),
                "outer_matched": families.get("box_outer", {}).get("matched", 0),
                "outer_reference": families.get("box_outer", {}).get("visible", 0),
                "peg_matched": peg["matched_3mm"],
                "peg_reference": peg["visible_reference"],
                "peg_unmatched": peg["predictions"] - peg["matched_3mm"],
                "box_unmatched": box["predictions"] - box["matched_3mm"],
            }
        episodes.append(
            {
                **{key: row[key] for key in ("seed", "mode", "frame")},
                "grasp": grasp,
                "scores": scores,
            }
        )
    return episodes


def evaluate(config):
    """Run the causal replay and store per-frame scores and aggregate counts."""
    root = Path(config["dataset"])
    source = root / "trajectory.json"
    rows = json.loads(source.read_text())["rows"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["mode"])].append(row)
    jobs = [(root, frames) for frames in groups.values()]
    with ExperimentLogger(
        team="graph",
        model="trajectory_rim_recovery",
        config=config,
        seed=20261114,
        job_type="evaluation",
    ) as logger:
        with ProcessPoolExecutor(
            max_workers=config["workers"], initializer=initialize
        ) as pool:
            results = []
            for frames in pool.map(score_rollout, jobs):
                results.extend(frames)
        totals = {
            name: Counter()
            for name in (
                "frame_only",
                "temporal",
                "consensus",
                "peg_size",
                "ranked_rim",
                "peg_silhouette",
                "depth_ranked_rim",
            )
        }
        for frame in results:
            for name, counts in totals.items():
                counts.update(frame["scores"][name])
        path = Path(config["output_root"]) / logger.run_id / "trajectory.json"
        path.parent.mkdir(parents=True, exist_ok=False)
        write_json(
            path,
            {"rows": results, "totals": totals, "source_sha256": file_sha256(source)},
        )
        logger.link_output(path)
        logger.record(frames=len(results), totals=totals)
        print(path, json.dumps(totals), flush=True)
