"""Checksummed full episodes and paired, pre-association synthetic corner removal."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import canonical_hash, file_sha256
from graph.temporal.corner_tracking.tracker import Observation


FORMAT = "observed_corner_tracks_v1"


@dataclass
class Frame:
    """Keep inference observations separate from certification and geometry targets."""

    sample_id: str
    observation: Observation
    physical_ids: np.ndarray
    reference_xyz: np.ndarray
    reference_kind: np.ndarray
    target_object21: np.ndarray


@dataclass
class Episode:
    name: str
    parent: str
    split: str
    frames: list
    eligibility_cache: dict = field(default_factory=dict, repr=False)


def write_episode(path, episode):
    """Pack variable node counts without changing any raw RGB-D source file."""
    frames = episode.frames
    count = max(1, max(len(frame.observation.xyz) for frame in frames))
    length = len(frames)
    xyz = np.zeros((length, count, 3), dtype=np.float32)
    kind = np.full((length, count), -1, dtype=np.int64)
    valid = np.zeros((length, count), dtype=bool)
    quality = np.zeros((length, count), dtype=np.float32)
    quality_valid = np.zeros((length, count), dtype=bool)
    identities = np.full((length, count), -1, dtype=np.int64)
    for time, frame in enumerate(frames):
        size = len(frame.observation.xyz)
        xyz[time, :size] = frame.observation.xyz
        kind[time, :size] = frame.observation.kind
        valid[time, :size] = True
        quality[time, :size] = frame.observation.confidence
        quality_valid[time, :size] = frame.observation.confidence_valid
        identities[time, :size] = frame.physical_ids
    np.savez_compressed(path, xyz=xyz, kind=kind, valid=valid, confidence=quality,
                        confidence_valid=quality_valid, physical_ids=identities,
                        robot=np.stack([frame.observation.robot for frame in frames]),
                        reference_xyz=np.stack([frame.reference_xyz for frame in frames]),
                        reference_kind=frames[0].reference_kind,
                        object21=np.stack([frame.target_object21 for frame in frames]),
                        sample_ids=np.asarray([frame.sample_id for frame in frames]))


def read_episode(path, row):
    """Reject shifted, incomplete or malformed episodes before model execution."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    xyz = arrays["xyz"]
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("Episode xyz must be [T,N,3]")
    length, count, _ = xyz.shape
    references = arrays["reference_xyz"]
    if references.ndim != 3 or references.shape[0] != length or references.shape[-1] != 3:
        raise ValueError("reference_xyz must be [T,P,3]")
    expected_shapes = dict(kind=(length, count), valid=(length, count),
                           confidence=(length, count), confidence_valid=(length, count),
                           physical_ids=(length, count), robot=(length, 9),
                           object21=(length, 21), reference_kind=(references.shape[1],),
                           sample_ids=(length,))
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Episode {name} must have shape {shape}")
    for name in ("xyz", "confidence", "robot", "reference_xyz", "object21"):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Episode {name} must be finite float32")
    for name in ("valid", "confidence_valid"):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"Episode {name} must be boolean")
    for name in ("kind", "physical_ids", "reference_kind"):
        if arrays[name].dtype != np.int64:
            raise ValueError(f"Episode {name} must be int64")
    ids = arrays["physical_ids"]
    if np.any((ids < -1) | (ids >= references.shape[1])):
        raise ValueError("physical_ids must be -1 or index a reference point")
    if length != row["frames"] or length < 2:
        raise ValueError("Need the declared complete episode with at least two frames")
    frames = []
    for time in range(length):
        sample_id = str(arrays["sample_ids"][time])
        if sample_id != f"{row['episode']}/frame_{time:06d}.npz":
            raise ValueError("Episodes must contain contiguous sample IDs starting at frame zero")
        keep = arrays["valid"][time]
        observation = Observation(xyz[time, keep], arrays["kind"][time, keep],
                                  arrays["confidence"][time, keep], arrays["confidence_valid"][time, keep],
                                  arrays["robot"][time])
        observation.validate()
        known = ids[time, keep] >= 0
        if np.any(arrays["reference_kind"][ids[time, keep][known]] != observation.kind[known]):
            raise ValueError("Certified reference object kind disagrees with detection")
        frames.append(Frame(sample_id, observation, ids[time, keep], references[time],
                            arrays["reference_kind"], arrays["object21"][time]))
    return Episode(row["episode"], row["parent"], row["split"], frames)


def load_dataset(root):
    """Validate hashes and parent separation; training never loads test episodes."""
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    content = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if manifest.get("format") != FORMAT or canonical_hash(content) != manifest.get("dataset_sha256"):
        raise ValueError("Corner dataset format/hash mismatch")
    if manifest.get("time_unit") != "saved_frame":
        raise ValueError("This corner dataset format requires time_unit=saved_frame")
    groups = {name: [] for name in ("train", "calibration", "validation", "test")}
    parent_splits, seen = {}, set()
    for row in manifest["episodes"]:
        split = row["split"]
        if split not in groups or row["episode"] in seen:
            raise ValueError("Unknown split or duplicate episode")
        if parent_splits.setdefault(row["parent"], split) != split:
            raise ValueError("A parent cannot occur in multiple splits")
        seen.add(row["episode"])
        if split == "test":
            continue
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
            raise ValueError("Episode path/checksum mismatch")
        groups[split].append(read_episode(path, row))
    if any(not groups[name] for name in ("train", "calibration", "validation")):
        raise ValueError("Need nonempty, parent-disjoint train/calibration/validation")
    return groups, manifest


def save_dataset(root, episodes, provenance):
    """Create a new dataset only; an existing destination is never overwritten."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, episode in enumerate(episodes):
        filename = f"episode_{index:06d}.npz"
        write_episode(root / filename, episode)
        rows.append(dict(episode=episode.name, parent=episode.parent, split=episode.split,
                         frames=len(episode.frames), file=filename, sha256=file_sha256(root / filename)))
    manifest = dict(format=FORMAT, time_unit="saved_frame", episodes=rows, provenance=provenance)
    manifest["dataset_sha256"] = canonical_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare(config):
    """Reuse full-frame family graphs, BOX certification and raw observed robot data.

    The family dataset must be contiguous; selected-frame cohorts are rejected.
    PEG IDs use the existing conservative coordinate/projection certification.
    This is label preparation, never an online tracker or detector replacement.
    """
    from data_collection.episode_dataset import read_frame
    from data_collection.physical_vertex_filter import peg_corner_ids
    from data_collection.vertex_family_manifest import read_manifest
    from training.gnn.data import load_cached_sample
    from graph.utils.object21_targets import object21_from_gt
    from visual.corner_detection.evaluation import corners
    from utils.experiment_logger import ExperimentLogger

    root = Path(config["source"]["family_dataset"]).expanduser().resolve()
    manifest = read_manifest(root)
    selection = manifest.get("selection", {})
    if "frames_per_parent" not in selection or selection["frames_per_parent"] is not None:
        raise ValueError("Source must explicitly declare full episodes with selection.frames_per_parent=null")
    fraction = config.get("calibration_fraction", 0.1)
    if type(fraction) not in (float, int) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("calibration_fraction must be finite and strictly between zero and one")
    parents = sorted({row["parent_id"] for row in manifest["samples"] if row["split"] == "train"})
    np.random.default_rng(config.get("seed", 0)).shuffle(parents)
    count = max(1, int(len(parents) * fraction))
    if count >= len(parents):
        raise ValueError("Need at least two training parents to reserve calibration")
    calibration_parents = set(parents[:count])
    grouped = {}
    for row in manifest["samples"]:
        episode = str(Path(row["sample_id"]).parent)
        grouped.setdefault(episode, []).append(row)
    episodes = []
    for name, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["sample_id"])
        expected = [f"{name}/frame_{time:06d}.npz" for time in range(len(rows))]
        if [row["sample_id"] for row in rows] != expected:
            raise ValueError(f"Need contiguous full episodes, not selected frames: {name}")
        frames = []
        for row in rows:
            paths = {}
            for key, digest in (("graph_path", "graph_sha256"), ("labels_path", "labels_sha256"),
                                ("raw_path", "source_sha256")):
                path = (root / row[key]).resolve()
                if not path.is_relative_to(root) or file_sha256(path) != row[digest]:
                    raise ValueError(f"Family source path/hash mismatch: {row['sample_id']}")
                paths[key] = path
            sample = load_cached_sample(paths["graph_path"], row["sample_identity"], row["sample_id"])
            graph = sample.graph
            keep = graph.node_valid[0].numpy()
            xyz = graph.xyz_world[0, keep].numpy()
            kind = graph.object_type[0, keep].numpy()
            observation, gt = read_frame(paths["raw_path"], config.get("camera", "base_camera"))
            np.testing.assert_allclose(sample.targets["object21"].numpy(), object21_from_gt(gt).numpy(),
                                       atol=1e-6, rtol=1e-5,
                                       err_msg="Cached Object21 target differs from the paired raw GT")
            with np.load(paths["raw_path"], allow_pickle=False) as raw:
                robot = np.concatenate((raw["observation/extra/tcp_pose"][0],
                                        raw["observation/agent/qpos"][0, -2:])).astype(np.float32)
            if not np.allclose(robot[:3], graph.tcp_xyz_world[0].numpy(), atol=1e-6, rtol=0):
                raise ValueError("Observed robot position differs from cached TCP")
            with np.load(paths["labels_path"], allow_pickle=False) as labels:
                np.testing.assert_array_equal(labels["xyz_world"], xyz)
                ids = np.full(len(xyz), -1, dtype=np.int64)
                peg_ids = peg_corner_ids(xyz, labels["uv"], kind, gt, observation)
                ids[kind == 0] = peg_ids[kind == 0]
                box_known = (kind == 1) & (labels["physical_id"] >= 0)
                # Supervision namespace: PEG 0..7, BOX exterior/rim 8..23.
                # These physical reference indices never become runtime IDs.
                ids[box_known] = labels["physical_id"][box_known] + 8
            references = corners(gt)
            obs = Observation(xyz, kind, graph.confidence[0, keep].numpy(),
                              graph.confidence_valid[0, keep].numpy(), robot)
            frames.append(Frame(row["sample_id"], obs, ids,
                                np.asarray([point["xyz"] for point in references], dtype=np.float32),
                                np.asarray([point["object_id"] for point in references], dtype=np.int64),
                                sample.targets["object21"][0].numpy()))
        split = rows[0]["split"]
        if any(row["split"] != split or row["parent_id"] != rows[0]["parent_id"] for row in rows):
            raise ValueError("An episode cannot change split or parent")
        if split == "train" and rows[0]["parent_id"] in calibration_parents:
            split = "calibration"
        episodes.append(Episode(name, rows[0]["parent_id"], split, frames))
    with ExperimentLogger(team="graph", config=config, seed=config.get("seed", 0),
                          model="corner_tracking_data", job_type="data_preparation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        packed = save_dataset(output, episodes, dict(source_dataset_sha256=manifest["dataset_sha256"],
                                                     config=config, certification="existing family BOX; PEG coordinate/projection"))
        logger.record(dataset_sha256=packed["dataset_sha256"])
        logger.link_output(output / "manifest.json")
        print(f"Corner tracking dataset: {output}", flush=True)
        return str(output)


def confirmed_eligibility(episode, calibration, settings):
    """Replay a common causal baseline before choosing backbone-independent masks."""
    from dataclasses import asdict
    from graph.temporal.corner_tracking.model import ConstantVelocityMotion
    from graph.temporal.corner_tracking.tracker import CornerTracker

    key = canonical_hash(dict(radii=calibration.radii.tolist(), settings=asdict(settings)))
    if key in episode.eligibility_cache:
        return episode.eligibility_cache[key]
    tracker = CornerTracker(ConstantVelocityMotion(), calibration, settings)
    identities, eligible = {}, []
    for frame in episode.frames:
        eligible.append({reference for identifier, reference in identities.items()
                         if reference >= 0 and identifier in tracker.tracks
                         and tracker.tracks[identifier].confirmed})
        # Eligibility depends only on causal associations, not on a GNN graph.
        result = tracker.step(frame.observation, include_graph=False)
        assignments = dict(result.assignments)
        assignments.update(result.births)
        for identifier, index in assignments.items():
            reference = int(frame.physical_ids[index])
            if reference >= 0:
                if identifier not in identities:
                    identities[identifier] = reference
                elif identities[identifier] != reference:
                    identities[identifier] = -1
    episode.eligibility_cache[key] = eligible
    return eligible


def mask_plan(episode, seed, probabilities=None, length_probabilities=None, eligible_references=None):
    """Choose one paired mask using prior certification only for eligibility.

    Plans are independent of either learned tracker. IDs never enter model input;
    an eligible corner has two certified observations in the three preceding frames.
    """
    rng = np.random.default_rng(seed)
    if probabilities is None:
        probabilities = [0.25, 0.40, 0.25, 0.10]
    if length_probabilities is None:
        length_probabilities = [0.20, 0.40, 0.30, 0.10]
    mode = str(rng.choice(["natural", "single", "group", "object"], p=probabilities))
    result = dict(episode=episode.name, mode=mode, start=0, stop=0, references=[], kind=None,
                  requested_length=0, actual_length=0)
    if mode == "natural":
        return result
    candidates = []
    for time in range(2, len(episode.frames)):
        counts = {}
        for frame in episode.frames[max(0, time - 3):time]:
            for identity in set(frame.physical_ids[frame.physical_ids >= 0]):
                counts[int(identity)] = counts.get(int(identity), 0) + 1
        eligible = [identity for identity, hits in counts.items() if hits >= 2]
        if eligible_references is not None:
            eligible = [identity for identity in eligible if identity in eligible_references[time]]
        kinds = episode.frames[time].reference_kind
        for kind in sorted(set(kinds[eligible])):
            group = [identity for identity in eligible if kinds[identity] == kind]
            if len(group) >= (2 if mode == "group" else 1):
                candidates.append((time, int(kind), group))
    if not candidates:
        return dict(result, skipped="no previously observed eligible corners")
    lower, upper = [(1, 2), (3, 8), (9, 16), (17, 32)][int(rng.choice(4, p=length_probabilities))]
    length = int(rng.integers(lower, upper + 1))
    full_length_candidates = [candidate for candidate in candidates
                              if candidate[0] + length <= len(episode.frames)]
    choices = full_length_candidates or candidates
    time, kind, group = choices[int(rng.integers(len(choices)))]
    if mode == "single":
        group = [int(rng.choice(group))]
    elif mode == "group":
        size = int(rng.integers(2, len(group) + 1))
        group = list(map(int, rng.choice(group, size=size, replace=False)))
    stop = min(len(episode.frames), time + length)
    return dict(result, start=time, stop=stop, references=sorted(group), kind=kind,
                requested_length=length, actual_length=stop - time, truncated=stop - time < length)


def masked_frame(frame, plan, time, duplicate_radius_m=0.003):
    """Erase all selected evidence, including near-duplicate uncertified points."""
    keep = np.ones(len(frame.physical_ids), dtype=bool)
    if plan["start"] <= time < plan["stop"]:
        if plan["mode"] == "object":
            keep &= frame.observation.kind != plan["kind"]
        else:
            for reference in plan["references"]:
                near = np.linalg.norm(frame.observation.xyz - frame.reference_xyz[reference], axis=-1)
                same_object = frame.observation.kind == frame.reference_kind[reference]
                keep &= ~((frame.physical_ids == reference) | (same_object & (near <= duplicate_radius_m)))
    # The original indices are supervision bookkeeping, not tracker inputs.
    return frame.observation.subset(keep), frame.physical_ids[keep], np.flatnonzero(keep)
