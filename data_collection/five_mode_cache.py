"""Create separate detector-corner and TAPNext++ surface caches from paired rollouts."""

import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_collection import POLICY_MODES, collection_lock
from data_collection.storage import write_json


CACHE_TYPES = ("corners", "surface")


def load_collection(root):
    """Check that every assigned protocol record has one archived five-mode bundle."""
    manifest = json.loads((root / "manifest.json").read_text())
    protocol_path = root / "protocol.json"
    if manifest["status"] != "COMPLETE" or file_sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError(f"Expected a complete collection with its frozen protocol: {root}")

    protocol = json.loads(protocol_path.read_text())
    start, stop = manifest["effective_config"].get("record_range", [0, len(protocol["records"])])
    position = start
    for receipt in manifest["archives"]:
        begin, end = receipt["record_range"]
        name = f"episodes_{begin:05d}_{end:05d}.tar.zst"
        if (
            begin != position or not begin < end <= stop or receipt["file"] != name
            or receipt["episodes"] != len(POLICY_MODES) * (end - begin)
        ):
            raise ValueError(f"Unexpected collection archive range or file: {receipt}")
        position = end
    if position != stop or manifest["episodes"] != len(POLICY_MODES) * (stop - start):
        raise ValueError(f"Collection coverage does not match the protocol: {root}")
    return manifest, protocol


def read_episode_index(batch, records):
    """Match archive paths, frame counts and split labels to the source protocol."""
    expected = {(row["seed"], mode): row["dataset_split"]
                for row in records for mode in POLICY_MODES}
    index = json.loads((batch / "episode_index.json").read_text())
    for case in index:
        key = case["seed"], case["policy_mode"]
        relative_path = Path(case["path"])
        if (
            expected.pop(key, None) != case["split"]
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 2
            or relative_path.parts[0] not in ("pilot", "moving", "recovery")
        ):
            raise ValueError(f"Unexpected episode in batch: {case}")
        path = batch / relative_path
        episode = json.loads((path / "episode.json").read_text())
        if (
            episode["seed"] != case["seed"]
            or episode["path"] != path.name
            or episode["observations"] != case["observations"]
            or episode["observations"] != episode["steps"] + 1
            or len(episode["frame_sha256"]) != episode["observations"]
        ):
            raise ValueError(f"Invalid source episode metadata: {path}")
        yield case, episode, path
    if expected:
        raise ValueError(f"Missing seed and mode combinations: {sorted(expected)}")


def validate_surface(arrays, frames, query_count):
    """Validate ragged surface output without requiring points in occluded frames."""
    offsets = arrays["surface_offsets"]
    ids = arrays["surface_track_id"]
    anchor = int(arrays["surface_anchor_frame"])
    if (
        not -1 <= anchor < frames
        or offsets.shape != (frames + 1,)
        or offsets[0] != 0 or offsets[-1] != len(ids)
        or np.any(np.diff(offsets) < 0)
        or np.any((ids < 0) | (ids >= query_count))
        or arrays["surface_query_xy"].shape != (0 if anchor < 0 else query_count, 2)
        or arrays["surface_uv"].shape != (len(ids), 2)
        or arrays["surface_world_xyz"].shape != (len(ids), 3)
        or not all(np.isfinite(arrays[name]).all() for name in
                   ("surface_query_xy", "surface_uv", "surface_world_xyz"))
    ):
        raise ValueError("Invalid surface track offsets, coordinates or query IDs")


def build_surface_arrays(frames, corners, tracker, detector, settings):
    """Build frame-aligned surface tracks with independent Peg and Box anchors."""
    from data_collection.detector_surface_cache import track_episode

    plan = settings.get("plan")
    count = settings["surface_queries"]
    if plan is None:
        tracked = track_episode(frames, corners, tracker, detector, settings)
        surface = {key: value for key, value in tracked.items() if key.startswith("surface_")}
        validate_surface(surface, len(frames), count)
        surface["frame_indices"] = corners["frame_indices"]
        return surface
    if plan not in ("PLAN_A", "PLAN_B"):
        raise ValueError("tracking.plan must be PLAN_A or PLAN_B")
    if count != 32 or settings["corner_exclusion_px"] != 2:
        raise ValueError("PLAN_A and PLAN_B require 32 surface queries per object and a 2 px corner exclusion")

    streams = [
        track_episode(frames, corners, tracker, detector, settings, object_id=object_id)
        for object_id in (0, 1)
    ]
    query_counts = [len(stream["surface_query_xy"]) for stream in streams]
    anchors = np.asarray([int(stream["surface_anchor_frame"]) for stream in streams])
    for stream, actual, anchor in zip(streams, query_counts, anchors, strict=True):
        if plan == "PLAN_B" and anchor >= 0 and actual < count:
            raise ValueError("PLAN_B must retain all 32 surface FPS queries")
        validate_surface(stream, len(frames), actual if plan == "PLAN_B" else count)

    per_frame = [np.diff(stream["surface_offsets"]) for stream in streams]
    order = np.argsort(np.concatenate([
        np.repeat(np.arange(len(frames)), counts) for counts in per_frame
    ]), kind="stable")
    object_ids = np.repeat(np.arange(2, dtype=np.int8), query_counts)
    first_anchor = min((anchor for anchor in anchors if anchor >= 0), default=-1)
    ids = np.concatenate((
        streams[0]["surface_track_id"], streams[1]["surface_track_id"] + query_counts[0],
    ))[order]
    surface = {
        key: np.concatenate([stream[key] for stream in streams])[order]
        for key in ("surface_uv", "surface_world_xyz")
    }
    surface.update(
        frame_indices=corners["frame_indices"],
        surface_offsets=np.r_[0, np.cumsum(per_frame[0] + per_frame[1])],
        surface_anchor_frame=np.asarray(first_anchor),
        surface_object_anchor_frames=anchors,
        surface_query_xy=np.concatenate([stream["surface_query_xy"] for stream in streams]),
        surface_query_object_id=object_ids,
        surface_track_id=ids,
        surface_object_id=object_ids[ids],
    )
    validate_surface(surface, len(frames), sum(query_counts))
    return surface


def cache_batch(source, target, receipt, records, detector, tracker, settings):
    """Extract one verified archive and commit both cache files per episode."""
    from data_collection.corner_detection_cache import cache_episode, write_episode
    from evaluation.tapnextpp_surface_tracking import read_frames

    archive = source / receipt["file"]
    if file_sha256(archive) != receipt["sha256"]:
        raise ValueError(f"Raw archive checksum mismatch: {archive}")
    batch_name = archive.name.removesuffix(".tar.zst")
    for kind in CACHE_TYPES:
        (target / kind / batch_name).mkdir(parents=True, exist_ok=True)

    saved = []
    with TemporaryDirectory(prefix="cache_extract_", dir=target) as temporary:
        batch = Path(temporary)
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", temporary], check=True)
        for case, episode, path in read_episode_index(batch, records):
            detection = cache_episode(detector, path.parent, episode)
            frames = read_frames({"source": path}, episode)
            surface = build_surface_arrays(frames, detection, tracker, detector, settings)
            name = f'{case["seed"]}_{case["policy_mode"]}.npz'
            files = {}
            for kind, arrays in zip(CACHE_TYPES, (detection, surface), strict=True):
                relative = Path(kind) / batch_name / name
                write_episode(target / relative, arrays)
                files[kind] = dict(file=relative.as_posix(), sha256=file_sha256(target / relative))
            saved.append(dict(case, files=files, corners=len(detection["corner_uv"]),
                              surface_points=len(surface["surface_track_id"]),
                              surface_anchor_frame=int(surface["surface_anchor_frame"])))
    return dict(raw_archive=receipt, episodes=saved)


def completed_batch(target, result, source_receipt):
    """Check the pinned archive receipt and both cached files before skipping work."""
    if result["raw_archive"] != source_receipt or len(result["episodes"]) != source_receipt["episodes"]:
        return False
    for episode in result["episodes"]:
        for kind in CACHE_TYPES:
            receipt = episode["files"][kind]
            path = target / receipt["file"]
            if not path.is_file() or file_sha256(path) != receipt["sha256"]:
                return False
    return True


def build(config):
    """Cache complete five-mode collection shards without retaining extracted RGB-D."""
    import cv2
    import torch

    from evaluation.tapnextpp_surface_tracking import load_tracker
    from utils.experiment_logger import ExperimentLogger
    from visual import corner_detection

    source = Path(config["source_root"]).resolve()
    target = Path(config["output_root"]).resolve()
    target.mkdir(parents=True, exist_ok=True)
    minimum_free_gib = config.get("minimum_free_gib", 10)
    if (
        type(minimum_free_gib) not in (int, float)
        or not np.isfinite(minimum_free_gib) or minimum_free_gib <= 0
    ):
        raise ValueError("minimum_free_gib must be a finite positive number")
    manifest, protocol = load_collection(source)
    settings = config["tracking"]
    detector_root = Path(corner_detection.__file__).parent
    identity = {
        "format": "five_mode_corner_and_surface_caches_v1",
        "source_protocol_sha256": manifest["protocol_sha256"],
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "checkpoint_sha256": file_sha256(settings["checkpoint"]),
        "detector_source_sha256": {path.relative_to(detector_root).as_posix(): file_sha256(path)
                                   for path in sorted(detector_root.rglob("*.py"))},
        "builder_sha256": file_sha256(__file__),
        "tracking": settings,
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "opencv": cv2.__version__, "numpy": np.__version__},
    }
    if settings.get("plan") is not None:
        tracking_sources = ["detector_surface_cache.py"]
        if settings["plan"] == "PLAN_B":
            tracking_sources.append("peg_edge_queries.py")
        identity["tracking_source_sha256"] = {
            name: file_sha256(Path(__file__).with_name(name)) for name in tracking_sources
        }
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with collection_lock(target), ExperimentLogger(team="visual", model="five_mode_cache",
                                                   config=config, job_type="prepare") as logger:
        run_cache(source, target, manifest, protocol, settings, identity,
                  corner_detection.TorchCornerDetector.main, load_tracker, logger,
                  minimum_free_gib=minimum_free_gib)


def run_cache(source, target, manifest, protocol, settings, identity, make_detector, make_tracker, logger,
              *, minimum_free_gib=10):
    """Keep a checksum receipt for every completed pair of caches."""
    identity_path = target / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(f"Cache inputs changed: {target}")
    write_json(identity_path, identity)
    detector = tracker = None
    results = []
    started = perf_counter()
    frames = 0
    for receipt in manifest["archives"]:
        batch_path = target / (receipt["file"].removesuffix(".tar.zst") + ".json")
        result = json.loads(batch_path.read_text()) if batch_path.exists() else None
        if result is None or not completed_batch(target, result, receipt):
            if shutil.disk_usage(target).free < minimum_free_gib * 1024**3:
                raise OSError(f"Expected {minimum_free_gib:g} GiB free for a source batch and its caches: {target}")
            if detector is None:
                detector = make_detector(device=settings["detector_device"])
                tracker = make_tracker(settings)
            start, stop = receipt["record_range"]
            result = cache_batch(source, target, receipt, protocol["records"][start:stop],
                                 detector, tracker, settings)
            write_json(batch_path, result)
        results.append(result)
        frames += sum(row["observations"] for row in result["episodes"])
        elapsed = perf_counter() - started
        logger.log({"frames": frames, "elapsed_seconds": elapsed}, step=len(results))
        print(f'{len(results)}/{len(manifest["archives"])} batches, {frames} frames, {elapsed:.1f}s',
              flush=True)

    output = target / "manifest.json"
    write_json(output, dict(identity, status="COMPLETE", frames=frames, batches=results))
    logger.link_output(output)
