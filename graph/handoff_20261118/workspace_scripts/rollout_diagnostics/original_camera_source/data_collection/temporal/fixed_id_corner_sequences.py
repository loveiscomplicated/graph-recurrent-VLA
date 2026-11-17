"""Per-corner sequences with canonical GT IDs and unchanged detected coordinates.

Each sample begins at that corner's first certified observation. Models receive
only past observations and current observed robot state. GT coordinates are
labels for missing frames, never input features. No tracker or GNN is imported.
"""

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import canonical_hash, file_sha256
from utils.experiment_logger import ExperimentLogger, write


FORMAT = "fixed_id_detected_corner_sequences_v1"
SOURCE_FORMAT = "observed_corner_tracks_v1"
FAMILIES = ("peg", "box_outer", "hole_rim")
SIGNS = tuple(product((-1, 1), repeat=3))


def corner_catalog():
    """Define IDs by object-local sign combinations, independent of world pose."""
    frames = ("peg", "box", "hole")
    scales = ("peg_half_sizes", "[peg_half_sizes.x] * 3", "[peg_half_sizes.x, hole_radius, hole_radius]")
    catalog = []
    for family_index, family in enumerate(FAMILIES):
        for local_id, signs in enumerate(SIGNS):
            catalog.append(dict(
                vertex_id=family_index * 8 + local_id,
                family=family,
                local_corner_id=local_id,
                coordinate_frame=frames[family_index],
                local_signs=list(signs),
                local_coordinate_rule=f"local_signs * {scales[family_index]}",
            ))
    return catalog


def read_source_manifest(root):
    """Accept the existing GT corner layout while preserving parent separation."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    content = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if manifest.get("format") != SOURCE_FORMAT or canonical_hash(content) != manifest.get("dataset_sha256"):
        raise ValueError("Source must be a checksummed observed_corner_tracks_v1 dataset")
    if manifest.get("time_unit") != "saved_frame":
        raise ValueError("Source time_unit must be saved_frame")
    parents, episodes = {}, set()
    for row in manifest["episodes"]:
        if row["split"] not in ("train", "calibration", "validation", "test"):
            raise ValueError(f"Unknown source split: {row['split']}")
        if row["episode"] in episodes:
            raise ValueError(f"Duplicate source episode: {row['episode']}")
        episodes.add(row["episode"])
        if parents.setdefault(row["parent"], row["split"]) != row["split"]:
            raise ValueError(f"Parent crosses dataset splits: {row['parent']}")
    return manifest


def load_source_episode(root, row):
    """Read only observed points, canonical GT trajectories and robot observations."""
    root = Path(root).resolve()
    path = (root / row["file"]).resolve()
    if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
        raise ValueError(f"Source episode checksum/path mismatch: {row['episode']}")
    names = ("xyz", "kind", "valid", "physical_ids", "reference_xyz", "reference_kind", "robot", "sample_ids")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in names}
    length = row["frames"]
    if type(length) is not int or length < 2:
        raise ValueError("Source episodes must contain at least two contiguous frames")
    xyz, targets, robot = arrays["xyz"], arrays["reference_xyz"], arrays["robot"]
    if xyz.ndim != 3 or xyz.shape[0] != length or xyz.shape[-1] != 3:
        raise ValueError(f"Source xyz must be [T,N,3]: {row['episode']}")
    if targets.shape != (length, 24, 3) or robot.shape != (length, 9):
        raise ValueError("Need canonical 24-corner GT [T,24,3] and observed TCP/finger state [T,9]")
    if not np.array_equal(arrays["reference_kind"], np.asarray([0] * 8 + [1] * 16)):
        raise ValueError("Source reference order must be PEG 0..7, BOX exterior 8..15, rim 16..23")
    for name in ("xyz", "reference_xyz", "robot"):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Source {name} must be finite float32")
    if arrays["valid"].shape != xyz.shape[:2] or arrays["valid"].dtype != np.bool_:
        raise ValueError("Source valid must be boolean [T,N]")
    identities = arrays["physical_ids"]
    if identities.shape != xyz.shape[:2] or identities.dtype != np.int64 or np.any((identities < -1) | (identities >= 24)):
        raise ValueError("Source physical_ids must be int64 [T,N], with UNKNOWN=-1 or canonical ID 0..23")
    known = arrays["valid"] & (identities >= 0)
    if (arrays["kind"].shape != xyz.shape[:2] or arrays["kind"].dtype != np.int64
            or np.any(arrays["kind"][known] != arrays["reference_kind"][identities[known]])):
        raise ValueError("Detected object kinds must agree with certified canonical IDs")
    expected = [f"{row['episode']}/frame_{index:06d}.npz" for index in range(length)]
    if arrays["sample_ids"].tolist() != expected:
        raise ValueError(f"Source episode is not a contiguous sequence: {row['episode']}")
    return arrays


def pack_episode(arrays):
    """Keep each ID only from its first actual detection, with no invented observations."""
    length = len(arrays["robot"])
    observed_xyz = np.zeros((length, 24, 3), dtype=np.float32)
    observed = np.zeros((length, 24), dtype=bool)
    unknown = 0
    duplicate = 0
    for frame in range(length):
        indices = np.flatnonzero(arrays["valid"][frame])
        ids = arrays["physical_ids"][frame, indices]
        unknown += int(np.sum(ids < 0))
        for vertex_id in np.unique(ids[ids >= 0]):
            candidates = indices[ids == vertex_id]
            # Preserve one actual detector value without averaging or snapping.
            # Oracle GT resolves rare duplicate certificates during annotation.
            errors = np.linalg.norm(arrays["xyz"][frame, candidates]
                                    - arrays["reference_xyz"][frame, vertex_id], axis=1)
            selected = candidates[int(np.argmin(errors))]
            observed_xyz[frame, vertex_id] = arrays["xyz"][frame, selected]
            observed[frame, vertex_id] = True
            duplicate += len(candidates) - 1
    seen_ids = np.flatnonzero(observed.any(axis=0)).astype(np.int64)
    vertex_ids = []
    terminal_only = 0
    starts = []
    offsets = [0]
    observations = []
    flags = []
    targets = []
    gt_targets = []
    prediction_valid = []
    statistics = []
    for vertex_id in seen_ids:
        start = int(np.flatnonzero(observed[:, vertex_id])[0])
        if start == length - 1:
            terminal_only += 1
            continue
        vertex_ids.append(int(vertex_id))
        visible = observed[start:, vertex_id]
        measurements = observed_xyz[start:, vertex_id]
        gt = arrays["reference_xyz"][start:, vertex_id]
        target = np.where(visible[:, None], measurements, gt)
        valid = np.ones(len(visible), dtype=bool)
        valid[0] = False  # First observation initializes history; it has no past.
        starts.append(start)
        offsets.append(offsets[-1] + len(visible))
        observations.append(measurements)
        flags.append(visible)
        targets.append(target)
        gt_targets.append(gt)
        prediction_valid.append(valid)
        statistics.append(dict(
            vertex_id=int(vertex_id),
            start_frame=start,
            frames=len(visible),
            predicted_visible=int(np.sum(valid & visible)),
            predicted_missing=int(np.sum(valid & ~visible)),
        ))
    stats = dict(
        unknown_detections=unknown,
        duplicate_certificates=duplicate,
        never_seen_vertices=24 - len(seen_ids),
        terminal_only_vertices=terminal_only,
        sequences=statistics,
    )
    if not vertex_ids:
        return None, stats
    packed = dict(
        vertex_ids=np.asarray(vertex_ids, dtype=np.int64),
        start_frames=np.asarray(starts, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        observed_xyz_world=np.concatenate(observations),
        observed=np.concatenate(flags),
        target_xyz_world=np.concatenate(targets),
        gt_xyz_world=np.concatenate(gt_targets),
        prediction_valid=np.concatenate(prediction_valid),
        robot_state=arrays["robot"],
        sample_ids=arrays["sample_ids"],
    )
    return packed, stats


def validate_packed_episode(arrays):
    """Reject missing, duplicated or noncausal sequence records before training."""
    ids, starts, offsets = arrays["vertex_ids"], arrays["start_frames"], arrays["offsets"]
    if (ids.dtype != np.int64 or ids.ndim != 1 or not len(ids)
            or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= 24))):
        raise ValueError("vertex_ids must be unique canonical int64 IDs 0..23")
    if starts.shape != ids.shape or offsets.shape != (len(ids) + 1,) or offsets[0] != 0:
        raise ValueError("Sequence start_frames/offsets must match vertex_ids")
    if starts.dtype != np.int64 or offsets.dtype != np.int64 or np.any(np.diff(offsets) <= 0):
        raise ValueError("Sequence offsets must be strictly increasing int64 indices")
    rows = int(offsets[-1])
    length = len(arrays["robot_state"])
    if np.any((starts < 0) | (starts >= length - 1)) or not np.array_equal(np.diff(offsets), length - starts):
        raise ValueError("Each corner sequence must extend from its first observation to episode end")
    for name in ("observed_xyz_world", "target_xyz_world", "gt_xyz_world"):
        value = arrays[name]
        if value.shape != (rows, 3) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite float32 [sequence_rows,3]")
    robot = arrays["robot_state"]
    if robot.shape != (length, 9) or robot.dtype != np.float32 or not np.isfinite(robot).all():
        raise ValueError("robot_state must be finite float32 [T,9]")
    for name in ("observed", "prediction_valid"):
        if arrays[name].shape != (rows,) or arrays[name].dtype != np.bool_:
            raise ValueError(f"{name} must be boolean [sequence_rows]")
    visible = arrays["observed"]
    if not np.all(visible[offsets[:-1]]) or np.any(arrays["prediction_valid"][offsets[:-1]]):
        raise ValueError("First sightings initialize history and must not be prediction targets")
    expected_valid = np.ones(rows, dtype=bool)
    expected_valid[offsets[:-1]] = False
    if not np.array_equal(arrays["prediction_valid"], expected_valid):
        raise ValueError("Every post-first-sighting frame must be a prediction target")
    if np.any(arrays["observed_xyz_world"][~visible] != 0):
        raise ValueError("Unobserved input coordinates must be zero")
    expected_targets = np.where(visible[:, None], arrays["observed_xyz_world"], arrays["gt_xyz_world"])
    if not np.array_equal(arrays["target_xyz_world"], expected_targets):
        raise ValueError("Visible targets must equal actual detections; missing targets must equal GT")


@dataclass(frozen=True)
class CornerSequence:
    """One independent sequence; the same temporal model is reused for every ID."""

    episode: str
    parent: str
    vertex_id: int
    start_frame: int
    inputs: np.ndarray
    targets: np.ndarray
    observed: np.ndarray
    prediction_valid: np.ndarray
    last_observed_xyz_world: np.ndarray
    target_delta_xyz: np.ndarray


def sequences_from_arrays(arrays, row):
    """Shift measurements by one frame so current visible targets cannot be copied."""
    for index, vertex_id in enumerate(arrays["vertex_ids"]):
        left, right = arrays["offsets"][index:index + 2]
        start = int(arrays["start_frames"][index])
        measured = arrays["observed_xyz_world"][left:right]
        visible = arrays["observed"][left:right]
        previous_xyz = np.zeros_like(measured)
        previous_visible = np.zeros((len(visible), 1), dtype=np.float32)
        previous_xyz[1:] = measured[:-1]
        previous_visible[1:, 0] = visible[:-1]
        robot = arrays["robot_state"][start:]
        inputs = np.concatenate((previous_xyz, previous_visible, robot), axis=1)
        targets = arrays["target_xyz_world"][left:right]

        # Reconstruct a predicted displacement from the last *past* detection.
        # The first row initializes the sequence and is never a prediction target.
        last_observed = np.empty_like(measured)
        most_recent = measured[0]
        for time in range(len(visible)):
            last_observed[time] = most_recent
            if visible[time]:
                most_recent = measured[time]
        yield CornerSequence(
            episode=row["episode"],
            parent=row["parent"],
            vertex_id=int(vertex_id),
            start_frame=start,
            inputs=inputs,
            targets=targets,
            observed=visible,
            prediction_valid=arrays["prediction_valid"][left:right],
            last_observed_xyz_world=last_observed,
            target_delta_xyz=targets - last_observed,
        )


def read_manifest(root):
    """Validate the completed dataset identity before returning any samples."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    content = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if manifest.get("format") != FORMAT or canonical_hash(content) != manifest.get("dataset_sha256"):
        raise ValueError("Fixed-ID sequence manifest format/checksum mismatch")
    return manifest


def iter_sequences(root, split):
    """Yield ID-separated samples without importing a tracker, GNN or temporal model."""
    root = Path(root).resolve()
    manifest = read_manifest(root)
    for row in manifest["episodes"]:
        if row["split"] != split:
            continue
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Fixed-ID sequence checksum/path mismatch: {row['episode']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        validate_packed_episode(arrays)
        yield from sequences_from_arrays(arrays, row)


def prepare(config):
    """Prepare natural-visibility detected sequences, preserving all original splits."""
    root = Path(config["source"]["corner_dataset"]).resolve()
    source = read_source_manifest(root)
    rows = [row for row in source["episodes"] if row["split"] != "test"]
    with ExperimentLogger(
        team="graph", config=config, seed=0, model="fixed_id_corner_sequences",
        job_type="data_preparation",
    ) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        summary = dict(
            source_episodes=len(rows), episodes_without_prediction=0,
            episodes={}, sequences={}, predicted_visible={}, predicted_missing={},
            unknown_detections=0, duplicate_certificates=0, never_seen_vertices=0,
            terminal_only_vertices=0,
            by_family={
                family: dict(sequences=0, predicted_visible=0, predicted_missing=0)
                for family in FAMILIES
            },
        )
        for index, row in enumerate(rows):
            packed, stats = pack_episode(load_source_episode(root, row))
            for name in ("unknown_detections", "duplicate_certificates", "never_seen_vertices", "terminal_only_vertices"):
                summary[name] += stats[name]
            if packed is not None:
                validate_packed_episode(packed)
                filename = f"episode_{index:06d}.npz"
                np.savez_compressed(output / filename, **packed)
                episodes.append(dict(
                    episode=row["episode"], parent=row["parent"], split=row["split"],
                    source_sha256=row["sha256"], file=filename,
                    sha256=file_sha256(output / filename), sequences=stats["sequences"],
                ))
                split = row["split"]
                summary["episodes"][split] = summary["episodes"].get(split, 0) + 1
                summary["sequences"][split] = summary["sequences"].get(split, 0) + len(stats["sequences"])
                for name in ("predicted_visible", "predicted_missing"):
                    summary[name][split] = summary[name].get(split, 0) + sum(x[name] for x in stats["sequences"])
                for sequence in stats["sequences"]:
                    family = summary["by_family"][FAMILIES[sequence["vertex_id"] // 8]]
                    family["sequences"] += 1
                    for name in ("predicted_visible", "predicted_missing"):
                        family[name] += sequence[name]
            else:
                summary["episodes_without_prediction"] += 1
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(rows):
                print(f"Prepared {index + 1}/{len(rows)} episodes", flush=True)
        contract = dict(
            observations="unchanged certified detector XYZ; no GT coordinate replacement",
            inputs="previous observed XYZ(3), previous observed flag(1), current robot state(9)",
            targets="visible: actual detector XYZ; missing after first sighting: GT XYZ",
            displacement="target minus last detection strictly before the prediction frame",
            excluded="never-seen vertices, pre-first-sighting frames, first-sighting prediction, terminal-only sightings",
            id_usage="sample identity only, not an input feature",
            synthetic_occlusion=False,
            visibility="certified detection availability; unknown certificates are not forced to an ID",
        )
        manifest = dict(
            format=FORMAT, time_unit="saved_frame", catalog=corner_catalog(), episodes=episodes,
            source_dataset_sha256=source["dataset_sha256"],
            source_manifest_sha256=file_sha256(root / "manifest.json"),
            preparation_source_sha256=file_sha256(Path(__file__)),
            config=config, contract=contract,
            reserved_test_episodes_unread=sum(row["split"] == "test" for row in source["episodes"]),
        )
        manifest["dataset_sha256"] = canonical_hash(manifest)
        write(output / "summary.json", summary)
        # Publish a completion manifest only after all episode files are saved.
        temporary = output / "manifest.json.tmp"
        write(temporary, manifest)
        temporary.replace(output / "manifest.json")
        for name in ("manifest.json", "summary.json"):
            logger.link_output(output / name)
        logger.record(dataset_sha256=manifest["dataset_sha256"], summary=summary,
                      observation_mode="actual_certified_detections", simulator_executed=False)
        print(f"Fixed-ID corner sequences: {output}", flush=True)
        return str(output)
