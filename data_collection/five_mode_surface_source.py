"""Read pinned train/validation inputs for an additive Five-mode surface cache."""

from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile

import numpy as np

from data_collection.episode_dataset import file_sha256


ALLOWED_SPLITS = ("train", "validation")
BASE_KINDS = ("corners", "surface", "targets")
POLICY_MODES = (
    "Fixed_Hole_expert", "Fixed_Hole_expert_random", "Fixed_Hole_expert_random_expert",
    "Fixed_Hole_random", "Moving_Hole_expert",
)
FRAME_KEYS = (
    "observation/sensor_data/base_camera/rgb",
    "observation/sensor_data/base_camera/depth",
    "observation/sensor_param/base_camera/intrinsic_cv",
    "observation/sensor_param/base_camera/extrinsic_cv",
)


def checked_path(path, suffix):
    """Reject known legacy-cache locations and unexpected input file types before I/O."""
    resolved = Path(path).resolve()
    forbidden_names = {"tapnext_state36.npz", "tapnext_recovery_state36.npz"}
    if ("shared_tracking_cache" in resolved.parts or resolved.name in forbidden_names
            or not resolved.name.endswith(suffix)):
        raise ValueError(f"Unexpected or forbidden surface-cache input: {resolved}")
    return resolved


def pinned_json(settings):
    """Read only a configured JSON whose bytes match its frozen release identity."""
    path = checked_path(settings["path"], ".json")
    if file_sha256(path) != settings["sha256"]:
        raise ValueError(f"Surface source JSON checksum differs: {path}")
    return path, json.loads(path.read_text())


def selected_records(manifest, requested=None):
    """Freeze existing assignments and reject any selected batch containing test data."""
    records = manifest["records"]
    if manifest.get("format") != "five_mode_supervised_v1" or len(records) != manifest["episodes"]:
        raise ValueError("Expected the frozen Five-mode supervised release")
    by_key, seed_splits, batch_splits = {}, {}, {}
    for record in records:
        key = record["seed"], record["policy_mode"]
        split = record["split"]
        batch = record["batch"]
        if (type(key[0]) is not int or key[0] < 0 or key[1] not in POLICY_MODES
                or key in by_key or split not in (*ALLOWED_SPLITS, "test")
                or seed_splits.setdefault(key[0], split) != split
                or type(record["frames"]) is not int or record["frames"] < 1
                or not isinstance(batch, str) or Path(batch).name != batch or batch in (".", "..")):
            raise ValueError(f"Invalid Five-mode episode identity or split: {key}")
        by_key[key] = record
        batch_splits.setdefault(batch, set()).add(split)
    wanted = None
    if requested is not None:
        wanted = {(row["seed"], row["policy_mode"]) for row in requested}
        if (not wanted or len(wanted) != len(requested) or not wanted <= by_key.keys()
                or any(by_key[key]["split"] not in ALLOWED_SPLITS for key in wanted)):
            raise ValueError("episodes must select unique train/validation episodes; test is reserved")
    result = [row for row in records if row["split"] in ALLOWED_SPLITS
              and (wanted is None or (row["seed"], row["policy_mode"]) in wanted)]
    if not result:
        raise ValueError("No train/validation episodes selected")
    if any(not batch_splits[row["batch"]] <= set(ALLOWED_SPLITS) for row in result):
        raise ValueError("Selected raw batches must contain no test episodes")
    return result


def release_member_bytes(manifest_path, manifest, receipt, kind, archives, stack):
    """Read one original ordinary NPZ member after location and checksum validation."""
    root = manifest_path.parent
    declared = {row["file"] for row in manifest["archives"]}
    path = checked_path(root / receipt["archive"], ".tar")
    member = PurePosixPath(receipt["member"])
    if (kind not in BASE_KINDS or receipt["archive"] not in declared or not path.is_relative_to(root)
            or member.is_absolute() or ".." in member.parts or member.parent.name != kind
            or member.suffix != ".npz"):
        raise ValueError(f"Unexpected supervised {kind} member: {receipt}")
    if path not in archives:
        archives[path] = stack.enter_context(tarfile.open(path, "r:"))
    info = archives[path].getmember(member.as_posix())
    if not info.isfile():
        raise ValueError(f"Expected an ordinary release NPZ member: {member}")
    with archives[path].extractfile(info) as stream:
        contents = stream.read()
    if hashlib.sha256(contents).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Original {kind} checksum differs: {member}")
    return contents


def release_tables(manifest_path, manifest, record, archives, stack):
    """Verify the three original members; decode targets only for frame-index alignment."""
    if record["split"] not in ALLOWED_SPLITS:
        raise ValueError("Cannot read test release members")
    tables = {}
    for kind in BASE_KINDS:
        receipt = record["files"][kind]
        contents = release_member_bytes(manifest_path, manifest, receipt, kind, archives, stack)
        with np.load(BytesIO(contents), allow_pickle=False) as arrays:
            if not np.array_equal(arrays["frame_indices"], np.arange(record["frames"])):
                raise ValueError(f"Original {kind} frame IDs differ: {receipt['member']}")
            if kind == "corners":
                tables[kind] = {name: arrays[name] for name in (
                    "frame_indices", "corner_offsets", "corner_uv",
                )}
    return tables["corners"]


def raw_archive(raw_root, records):
    """Resolve only one selected batch, checking its receipt before decompression."""
    if any(row["split"] not in ALLOWED_SPLITS for row in records):
        raise ValueError("Cannot open a test raw archive")
    batches = {row["batch"] for row in records}
    digests = {row["raw_sha256"] for row in records}
    if len(batches) != 1 or len(digests) != 1:
        raise ValueError("Raw archive rows must share one batch and checksum")
    directory = (raw_root / records[0]["batch"]).resolve()
    if not directory.is_relative_to(raw_root):
        raise ValueError("Raw batch escapes raw_root")
    paths = list(directory.glob("*.tar.zst"))
    if len(paths) != 1:
        raise ValueError(f"Expected one raw .tar.zst in {directory}")
    path = checked_path(paths[0], ".tar.zst")
    if not path.is_relative_to(raw_root) or file_sha256(path) != next(iter(digests)):
        raise ValueError(f"Raw archive location or checksum differs: {path}")
    return path


@contextmanager
def open_raw_stream(path):
    """Always stop the decompressor when a consumer fails or closes a generator early."""
    process = subprocess.Popen(["zstd", "-qdc", str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            yield archive
        while process.stdout.read(65536):
            pass
        if process.wait() != 0:
            raise ValueError(f"Raw archive decompression failed: {path}")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def _raw_member_name(member):
    """Reject unsafe TAR paths before interpreting or staging their contents."""
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
        raise ValueError(f"Unsafe raw TAR member: {name}")
    return name


def _raw_episode_directories(path, expected):
    """Find and validate the index even when it follows the raw frame members."""
    directories = {}
    cases = None
    with open_raw_stream(path) as archive:
        for member in archive:
            name = _raw_member_name(member)
            if name == PurePosixPath("episode_index.json") and member.isfile():
                if cases is not None:
                    raise ValueError("Duplicate raw episode index")
                with archive.extractfile(member) as stream:
                    cases = json.load(stream)
    if cases is None:
        raise ValueError(f"Raw archive is missing its episode index: {path}")
    if any(row["split"] not in ALLOWED_SPLITS for row in cases):
        raise ValueError("Raw batch contains test episodes; refusing to process it")
    for case in cases:
        key = case["seed"], case["policy_mode"]
        if key not in expected:
            continue
        row = expected[key]
        directory = PurePosixPath(case["path"])
        if (directory.is_absolute() or ".." in directory.parts or len(directory.parts) != 2
                or directory.parts[0] not in ("pilot", "moving", "recovery")
                or directory in directories or key in directories.values()
                or case["observations"] != row["frames"] or case["split"] != row["split"]):
            raise ValueError(f"Raw index disagrees with frozen release: {key}")
        directories[directory] = key
    if set(directories.values()) != expected.keys():
        raise ValueError("Raw index omits selected episodes")
    return directories


def raw_episodes(path, records, scratch):
    """Stage verified selected episodes independently of TAR member order.

    A metadata-only first pass locates the index and rejects test batches before
    any episode payload is opened. The second pass accepts receipts and frames
    in either order; every frame hash is verified before yielding an episode.
    """
    expected = {(row["seed"], row["policy_mode"]): row for row in records}
    directories = _raw_episode_directories(path, expected)
    metadata, seen, pending_hashes, completed = {}, {}, {}, set()
    for directory, key in directories.items():
        (scratch / directory).mkdir(parents=True, exist_ok=False)
        seen[key] = set()
        pending_hashes[key] = {}
    with open_raw_stream(path) as archive:
        for member in archive:
            name = _raw_member_name(member)
            key = directories.get(name.parent)
            if key is None or not member.isfile():
                continue
            row = expected[key]
            destination = scratch / name.parent
            if name.name == "episode.json":
                with archive.extractfile(member) as stream:
                    episode = json.load(stream)
                if (key in metadata or episode["seed"] != key[0] or episode["path"] != name.parent.name
                        or episode["observations"] != row["frames"]
                        or len(episode["frame_sha256"]) != row["frames"]):
                    raise ValueError(f"Raw episode receipt disagrees with release: {key}")
                metadata[key] = episode
                for frame, digest in pending_hashes[key].items():
                    if digest != episode["frame_sha256"][frame]:
                        raise ValueError(f"Raw RGB-D frame checksum differs: {name.parent}/frame_{frame:06d}.npz")
                pending_hashes[key].clear()
            else:
                match = re.fullmatch(r"frame_(\d{6})\.npz", name.name)
                if match is None:
                    continue
                frame = int(match[1])
                if frame >= row["frames"] or frame in seen[key] or key in completed:
                    raise ValueError(f"Invalid or duplicate raw frame member: {name}")
                with archive.extractfile(member) as stream:
                    contents = stream.read()
                digest = hashlib.sha256(contents).hexdigest()
                if key in metadata:
                    if digest != metadata[key]["frame_sha256"][frame]:
                        raise ValueError(f"Raw RGB-D frame checksum differs: {name}")
                else:
                    pending_hashes[key][frame] = digest
                (destination / name.name).write_bytes(contents)
                seen[key].add(frame)
            if key in metadata and len(seen[key]) == row["frames"]:
                completed.add(key)
                yield row, destination
                shutil.rmtree(destination)
    if completed != expected.keys():
        raise ValueError(f"Raw archive is missing requested episode frames: {path}")


def read_rgbd(path):
    """Decode observation RGB-D and CV calibration only; never decode embedded GT."""
    with np.load(path, allow_pickle=False) as arrays:
        rgb, depth, intrinsic, extrinsic = (arrays[key][0] for key in FRAME_KEYS)
    depth = depth.squeeze(-1) if depth.ndim == 3 else depth
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8
            or depth.shape != rgb.shape[:2] or depth.dtype != np.int16
            or intrinsic.shape != (3, 3) or extrinsic.shape != (3, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
            or abs(np.linalg.det(intrinsic)) < 1e-12):
        raise ValueError(f"Expected aligned uint8 RGB, int16 depth and finite CV calibration: {path}")
    return rgb, depth, intrinsic, extrinsic
