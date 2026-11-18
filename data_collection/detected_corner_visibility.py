"""Classify visibility from the detector's merged corners in each frame."""

from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np


CONDITIONS = ("condition_1", "condition_2", "condition_3")
SPLITS = ("train", "validation", "test")
CORNER_FIELDS = (
    "frame_indices", "corner_offsets", "corner_world_xyz",
    "corner_object_id", "corner_contour_hole",
)


def coplanarity_tolerance(value):
    """Validate the minimum RMS distance to a plane, in metres."""
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("coplanarity_tolerance_m must be positive and finite")
    return tolerance


def _classify_validated_frame(xyz, kinds, rim, tolerance_m):
    """Apply the condition thresholds to already-validated detector arrays."""
    peg = kinds == 0
    peg_count = np.count_nonzero(peg)
    rim_count = np.count_nonzero((kinds == 1) & rim)
    if peg_count >= 4 and rim_count >= 3:
        peg_xyz = xyz[peg].astype(np.float64)
        centered = peg_xyz - peg_xyz.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        plane_rms_m = singular_values[-1] / np.sqrt(peg_count)
        if plane_rms_m > tolerance_m:
            return "condition_1"
    if peg_count >= 2 and rim_count >= 1:
        return "condition_2"
    return "condition_3"


def classify_frame(xyz, kinds, rim, *, tolerance_m):
    """Classify one frame from its current detector corners only.

    ``xyz`` and ``kinds`` contain every merged detector corner in the frame.
    ``rim`` identifies which Box corners belong to the Hole rim. TAPNext++
    surface points are deliberately absent from this interface.
    """
    xyz = np.asarray(xyz)
    kinds = np.asarray(kinds)
    rim = np.asarray(rim)
    tolerance = coplanarity_tolerance(tolerance_m)
    count = len(kinds)
    if (
        xyz.shape != (count, 3) or not np.isfinite(xyz).all()
        or kinds.shape != (count,) or kinds.dtype.kind not in "iu"
        or not np.isin(kinds, (0, 1)).all()
        or rim.shape != (count,) or rim.dtype != np.bool_
    ):
        raise ValueError("Invalid current-frame detector corners")
    return _classify_validated_frame(xyz, kinds, rim, tolerance)


def classify_episode(corners, frame_count, *, tolerance_m):
    """Return original frame numbers grouped by detector-corner condition."""
    tolerance = coplanarity_tolerance(tolerance_m)
    frames, offsets, xyz, kinds, rim = (corners[name] for name in CORNER_FIELDS)
    points = len(kinds)
    if (
        frames.dtype.kind not in "iu" or not np.array_equal(frames, np.arange(frame_count))
        or offsets.shape != (frame_count + 1,) or offsets.dtype.kind not in "iu"
        or offsets[0] != 0 or offsets[-1] != points
        or np.any(offsets[1:] < offsets[:-1])
        or xyz.shape != (points, 3) or not np.isfinite(xyz).all()
        or kinds.shape != (points,) or kinds.dtype.kind not in "iu"
        or not np.isin(kinds, (0, 1)).all()
        or rim.shape != (points,) or rim.dtype != np.bool_
    ):
        raise ValueError("Invalid per-frame detector corner arrays")

    groups = {condition: [] for condition in CONDITIONS}
    for frame, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        condition = _classify_validated_frame(
            xyz[start:stop], kinds[start:stop], rim[start:stop], tolerance,
        )
        groups[condition].append(frame)
    return groups


def validate_selection(manifest, expected_seeds, expected_episodes):
    """Check the complete selected cohort and preserve its frozen seed splits."""
    episodes = manifest["episodes"]
    seed_splits = {}
    seed_modes = defaultdict(set)
    frames = Counter()
    episode_counts = Counter()
    for episode in episodes:
        seed, mode, split = (episode[key] for key in ("seed", "policy_mode", "split"))
        count = episode["frames"]
        if (
            type(seed) is not int or not isinstance(mode, str) or split not in SPLITS
            or mode in seed_modes[seed] or seed_splits.setdefault(seed, split) != split
            or type(count) is not int or count < 1
        ):
            raise ValueError(f"Invalid selected episode: {seed}, {mode}, {split}")
        seed_modes[seed].add(mode)
        frames[split] += count
        episode_counts[split] += 1
    if (
        len(episodes) != expected_episodes or len(seed_modes) != expected_seeds
        or len({frozenset(modes) for modes in seed_modes.values()}) != 1
        or len(next(iter(seed_modes.values()))) != 5
        or manifest["seeds"] != expected_seeds
        or any(frames[split] != manifest["frames_by_split"].get(split, 0) for split in SPLITS)
        or any(episode_counts[split] != manifest["episodes_by_split"].get(split, 0) for split in SPLITS)
        or dict(Counter(seed_splits.values())) != manifest["seeds_by_split"]
    ):
        raise ValueError("Selected cache does not cover the expected five-mode cohort")


def read_cached_corners(stack, archives, root, episode):
    """Read only the manifest's checksum-pinned detector NPZ member."""
    receipt = episode["files"]["corners"]
    archive_path = (root / receipt["archive"]).resolve()
    member_path = PurePosixPath(receipt["member"])
    expected_name = f'{episode["seed"]}_{episode["policy_mode"]}.npz'
    if (
        not archive_path.is_relative_to(root) or archive_path.suffix != ".tar"
        or len(member_path.parts) != 3 or member_path.parts[0] != "corners"
        or ".." in member_path.parts or member_path.name != expected_name
    ):
        raise ValueError(f"Expected a Five-mode detector-corner member: {receipt}")
    if archive_path not in archives:
        archives[archive_path] = stack.enter_context(tarfile.open(archive_path, "r:"))
    archive = archives[archive_path]
    member = archive.getmember(receipt["member"])
    if not member.isfile():
        raise ValueError(f"Expected a regular detector-corner file: {member.name}")
    with archive.extractfile(member) as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Detector-corner checksum mismatch: {member.name}")
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in CORNER_FIELDS}


def run(config):
    """Count and index all frames of a selected Five-mode detector cache."""
    source = Path(config["selection_manifest"])
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Choose a new visibility index path: {output}")
    tolerance = coplanarity_tolerance(config["coplanarity_tolerance_m"])
    manifest_bytes = source.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["format"] != "five_mode_paired_cache_selection":
        raise ValueError(f"Expected a Five-mode cache selection manifest: {source}")
    validate_selection(manifest, config["expected_seeds"], config["expected_episodes"])
    cache_root = (source.parent / manifest["cache_root"]).resolve()
    counts = {condition: dict.fromkeys(SPLITS, 0) for condition in CONDITIONS}
    rows = []
    with ExitStack() as stack:
        archives = {}
        for episode in manifest["episodes"]:
            corners = read_cached_corners(stack, archives, cache_root, episode)
            try:
                groups = classify_episode(corners, episode["frames"], tolerance_m=tolerance)
            except ValueError as error:
                raise ValueError(f'{episode["files"]["corners"]["member"]}: {error}') from error
            split = episode["split"]
            for condition, frames in groups.items():
                counts[condition][split] += len(frames)
            row = {key: episode[key] for key in ("seed", "policy_mode", "split")}
            rows.append(dict(row, frame_indices=groups))

    result = {
        "format": "detected_corner_visibility_v2",
        "selection_manifest": source.as_posix(),
        "selection_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "effective_config": config,
        "seeds": manifest["seeds"],
        "episodes_by_split": manifest["episodes_by_split"],
        "frames_by_split": manifest["frames_by_split"],
        "counts": counts,
        "totals": {condition: sum(by_split.values()) for condition, by_split in counts.items()},
        "episodes": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("seeds", "episodes_by_split", "counts", "totals")},
                     indent=2), flush=True)
    print(f"Index: {output}", flush=True)
