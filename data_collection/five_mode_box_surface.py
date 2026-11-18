"""Append Box/Hole non-corner surface tracks without rewriting the Five-mode release."""

from collections import Counter, defaultdict
from contextlib import ExitStack, closing, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from data_collection.corner_detection_cache import write_episode
from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.five_mode_surface_source import (
    ALLOWED_SPLITS, checked_path, pinned_json, raw_archive, raw_episodes,
    read_rgbd, release_tables, selected_records,
)
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection.candidates import material_masks


def validate_tracking_settings(settings):
    """Reject precision/Graph combinations that cannot use the selected backend."""
    flags = ("tap_adaptive_crop", "tap_cuda_autocast", "tap_cache_cuda_autocast", "tap_cuda_graph")
    for name in flags:
        if type(settings.get(name, False)) is not bool:
            raise ValueError(f"tracking.{name} must be boolean")
    requested_device = settings.get("device", "cpu")
    if requested_device == "mlx":
        device_type = "mlx"
    else:
        try:
            device_type = torch.device(requested_device).type
        except (TypeError, RuntimeError, ValueError) as error:
            raise ValueError("tracking.device must name a Torch device or mlx") from error
    autocast = settings.get("tap_cuda_autocast", False)
    if autocast and device_type != "cuda":
        raise ValueError("tracking.tap_cuda_autocast requires a CUDA tracking device")
    if settings.get("tap_cuda_graph", False) and (
        device_type != "cuda" or not settings.get("tap_adaptive_crop", False)
    ):
        raise ValueError("tracking.tap_cuda_graph requires adaptive CUDA tracking")
    if settings.get("tap_cache_cuda_autocast", False) and (device_type != "cuda" or not autocast):
        raise ValueError("tracking.tap_cache_cuda_autocast requires CUDA FP16 autocast")


def prepare(config):
    """Validate the selected inventory and output locations without opening raw archives."""
    if config.get("splits") != list(ALLOWED_SPLITS):
        raise ValueError("Box surface generation requires splits: [train, validation]; test is reserved")
    if config.get("mode", "build") not in ("build", "preflight"):
        raise ValueError("mode must be build or preflight")
    validate_tracking_settings(config["tracking"])
    manifest_path, manifest = pinned_json(config["manifest"])
    index_path, _ = pinned_json(config["condition_index"])
    records = selected_records(manifest, config.get("episodes"))
    counts = {split: dict(episodes=sum(row["split"] == split for row in records),
                          frames=sum(row["frames"] for row in records if row["split"] == split))
              for split in ALLOWED_SPLITS}
    if counts != config["expected_counts"]:
        raise ValueError(f"Selected train/validation inventory differs from expected_counts: {counts}")
    raw_root = Path(config["raw_root"]).resolve()
    output = Path(config["output_root"]).resolve()
    scratch = Path(config["scratch_root"]).resolve()
    for writable in (output, scratch):
        if (writable.is_relative_to(manifest_path.parent)
                or manifest_path.parent.is_relative_to(writable)
                or writable.is_relative_to(raw_root) or raw_root.is_relative_to(writable)):
            raise ValueError("output_root and scratch_root must be separate from the source release/raw directories")
    settings = config["tracking"]
    for name in ("surface_queries", "tap_input_resolution"):
        if type(settings.get(name)) is not int or settings[name] < 1:
            raise ValueError(f"tracking.{name} must be an integer >= 1")
    radius = settings.get("corner_exclusion_px")
    if type(radius) not in (int, float) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("tracking.corner_exclusion_px must be positive and finite")
    if type(config.get("resume")) is not bool:
        raise ValueError("resume must be boolean")
    if type(config.get("export_candidate")) is not bool:
        raise ValueError("export_candidate must be boolean")
    if type(config.get("episodes_per_shard")) is not int or config["episodes_per_shard"] < 1:
        raise ValueError("episodes_per_shard must be an integer >= 1")
    free_gib = config.get("minimum_free_gib")
    if type(free_gib) not in (int, float) or not np.isfinite(free_gib) or free_gib < 0:
        raise ValueError("minimum_free_gib must be nonnegative and finite")
    checked_path(settings["checkpoint"], ".pt")
    if not shutil.which("zstd"):
        raise ValueError("The zstd executable is required for raw RGB-D archives")
    return manifest_path, manifest, index_path, records, counts


def box_mask(rgb):
    """Use the current detector's RGB material-mask algorithm without rerunning corners."""
    masks, _ = material_masks(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), optimized=True)
    return masks[1]


def separated_from_corners(xy, corners, radius):
    """Exclude both subpixel track locations and their sampled integer depth pixels."""
    xy = np.asarray(xy)
    corners = np.asarray(corners)
    samples = (xy, np.rint(xy))
    keep = np.ones(len(xy), dtype=bool)
    for points in samples:
        squared = ((points[:, None] - corners[None]) ** 2).sum(axis=-1)
        keep &= (squared > radius ** 2).all(axis=1)
    return keep


def tracking_cache_scope(tracker, settings):
    """Retain FP16 weight casts across frames without autocasting preprocessing.

    The tracker enables CUDA FP16 inside each call. A disabled outer autocast
    scope keeps the cache alive until the episode ends, matching the measured
    Peg replay. Inference mode bypasses weight caching, so use no_grad instead.
    """
    if not settings.get("tap_cache_cuda_autocast", False):
        return nullcontext()
    device = getattr(tracker, "device", None)
    if getattr(device, "type", None) != "cuda":
        raise ValueError("CUDA FP16 weight caching requires a CUDA tracker instance")
    if torch.is_inference_mode_enabled():
        raise ValueError("CUDA FP16 weight caching requires no_grad, not an enclosing inference_mode")
    return torch.autocast("cuda", enabled=False, cache_enabled=True)


@torch.no_grad()
def track_box_episode(directory, corners, tracker, settings):
    """Track one object stream; every emitted point has current Box-mask/depth support."""
    validate_tracking_settings(settings)
    with tracking_cache_scope(tracker, settings):
        return _track_box_frames(directory, corners, tracker, settings)


def _track_box_frames(directory, corners, tracker, settings):
    """Replay one complete episode inside the caller's precision/cache lifetime."""
    from evaluation.tapnextpp_surface_tracking import choose_points, lift, track_surface_frame

    frame_ids = corners["frame_indices"]
    offsets = corners["corner_offsets"]
    uv = corners["corner_uv"]
    if (frame_ids.dtype.kind not in "iu" or not np.array_equal(frame_ids, np.arange(len(frame_ids)))
            or offsets.shape != (len(frame_ids) + 1,) or offsets.dtype.kind not in "iu"
            or offsets[0] != 0 or offsets[-1] != len(uv) or np.any(np.diff(offsets) < 0)
            or uv.shape != (len(uv), 2) or not np.isfinite(uv).all()):
        raise ValueError("Invalid preserved detector corner frame layout")
    radius = settings["corner_exclusion_px"]
    count = settings["surface_queries"]
    state, anchor = None, -1
    queries = np.empty((0, 2), dtype=np.float32)
    emitted_uv, emitted_xyz, emitted_ids = [], [], []
    output_offsets = [0]
    for frame in range(len(frame_ids)):
        rgb, depth, intrinsic, extrinsic = read_rgbd(directory / f"frame_{frame:06d}.npz")
        mask = box_mask(rgb)
        support = mask & (depth > 0) & (depth < 4000)
        observed_corners = uv[slice(*offsets[frame:frame + 2])]
        initial_queries = None
        if state is None:
            candidate_depth = np.where(support, depth, 0)
            y, x = np.nonzero(candidate_depth)
            pixels = np.column_stack((x, y))
            keep = separated_from_corners(pixels, observed_corners, radius)
            candidate_depth[y[~keep], x[~keep]] = 0
            try:
                initial_queries = choose_points(mask, candidate_depth, count)
            except ValueError:
                output_offsets.append(output_offsets[-1])
                continue
            queries = initial_queries.copy()
            anchor = frame
        xy, visible, state = track_surface_frame(
            tracker, settings, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            query=initial_queries, state=state, observation_mask=mask,
        )
        xy, visible = np.asarray(xy), np.asarray(visible)
        if xy.shape != (count, 2) or visible.shape != (count,) or visible.dtype != np.bool_:
            raise ValueError(f"Invalid Box tracker output at frame {frame}")
        usable = visible & np.isfinite(xy).all(axis=1)
        safe_xy = np.where(np.isfinite(xy), xy, 0)
        usable &= separated_from_corners(safe_xy, observed_corners, radius)
        ids, camera_xyz = lift(safe_xy, depth, support, intrinsic, usable)
        world_xyz = ((camera_xyz - extrinsic[:, 3]) @ extrinsic[:, :3]).astype(np.float32)
        if not np.isfinite(world_xyz).all():
            raise ValueError(f"Box projection returned nonfinite world coordinates at frame {frame}")
        emitted_uv.append(xy[ids].astype(np.float32))
        emitted_xyz.append(world_xyz)
        emitted_ids.append(ids.astype(np.int32))
        output_offsets.append(output_offsets[-1] + len(ids))
    ids = np.concatenate(emitted_ids) if emitted_ids else np.empty(0, dtype=np.int32)
    return dict(
        frame_indices=frame_ids.copy(), surface_offsets=np.asarray(output_offsets, dtype=np.int64),
        surface_uv=np.concatenate(emitted_uv) if emitted_uv else np.empty((0, 2), dtype=np.float32),
        surface_world_xyz=np.concatenate(emitted_xyz) if emitted_xyz else np.empty((0, 3), dtype=np.float32),
        surface_track_id=ids, surface_object_id=np.ones(len(ids), dtype=np.int8),
        surface_anchor_frame=np.asarray(anchor, dtype=np.int64), surface_query_xy=queries,
        surface_query_object_id=np.ones(len(queries), dtype=np.int8),
    )


def cached_record(output, record, identity, resume):
    """Reuse only a committed episode whose source identity and bytes still match."""
    name = f'{record["seed"]}_{record["policy_mode"]}'
    receipt_path = output / "receipts" / f"{name}.json"
    path = output / "box_surface" / f"{name}.npz"
    if not receipt_path.exists():
        if path.exists():
            raise ValueError(f"Uncommitted Box cache exists; choose a new output or inspect it: {path}")
        return None
    if not resume:
        raise FileExistsError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (receipt["source_record_sha256"] != canonical_hash(record)
            or receipt["cache_identity_sha256"] != canonical_hash(identity)
            or receipt["box_surface"]["file"] != path.relative_to(output).as_posix()
            or file_sha256(path) != receipt["box_surface"]["sha256"]):
        raise ValueError(f"Committed Box cache identity or checksum differs: {receipt_path}")
    return dict(deepcopy(record), box_surface=receipt["box_surface"])


def build(config, *, tracker_factory=None):
    """Generate a resumable train/validation overlay; injectable trackers are smoke-only."""
    manifest_path, manifest, index_path, records, counts = prepare(config)
    if config.get("mode") == "preflight":
        print(f"Validated train/validation inventory: {counts}; no raw or tracker inputs read", flush=True)
        return counts
    # The existing collection lock imports simulator workflows; preflight and
    # the RGB-D helpers remain independent of that optional runtime.
    from data_collection.five_mode_collection import collection_lock
    settings = config["tracking"]
    checkpoint = checked_path(settings["checkpoint"], ".pt")
    if file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("TAPNext++ checkpoint checksum differs")
    source_root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), Path(__file__).with_name("five_mode_surface_source.py"),
               Path(__file__).with_name("five_mode_surface_export.py"),
               source_root / "visual/corner_detection/candidates.py",
               source_root / "visual/tapnextpp_cuda.py",
               source_root / "visual/tapnextpp_window.py",
               source_root / "evaluation/tapnextpp_surface_tracking.py"]
    identity = dict(
        format="five_mode_box_surface_overlay_v1",
        manifest_sha256=config["manifest"]["sha256"],
        condition_index_sha256=config["condition_index"]["sha256"],
        checkpoint_sha256=config["checkpoint_sha256"], tracking=settings,
        selected_records_sha256=canonical_hash(records), production_tracking=tracker_factory is None,
        export_candidate=config["export_candidate"], episodes_per_shard=config["episodes_per_shard"],
        sources={str(path.relative_to(source_root)): file_sha256(path) for path in sources},
        runtime=dict(numpy=np.__version__, opencv=cv2.__version__, torch=str(torch.__version__)),
    )
    output = Path(config["output_root"]).resolve()
    scratch_root = Path(config["scratch_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    with collection_lock(output), ExperimentLogger(
        team="visual", model="five_mode_box_surface", config=config, job_type="cache",
    ) as logger:
        identity_path = output / "identity.json"
        if identity_path.exists():
            if not config["resume"] or json.loads(identity_path.read_text()) != identity:
                raise ValueError("Output belongs to a different Box surface build, or resume is disabled")
        else:
            if any(output.iterdir()):
                # The writer lock itself is created before this empty-directory check.
                unexpected = [path for path in output.iterdir() if path.name != ".collection.lock"]
                if unexpected:
                    raise ValueError("Choose an empty output directory for a new Box surface build")
            write_json(identity_path, identity)
        completed, pending = {}, defaultdict(list)
        for record in records:
            key = record["seed"], record["policy_mode"]
            reused = cached_record(output, record, identity, config["resume"])
            if reused is None:
                pending[record["batch"]].append(record)
            else:
                completed[key] = reused
        tracker = None
        if pending:
            if tracker_factory is None:
                from evaluation.tapnextpp_surface_tracking import load_tracker

                tracker_factory = load_tracker
            tracker = tracker_factory(settings)
        (output / "box_surface").mkdir(exist_ok=True)
        (output / "receipts").mkdir(exist_ok=True)
        logger.record(cache_identity_sha256=canonical_hash(identity), counts=counts,
                      production_tracking=identity["production_tracking"], test_samples_read=False,
                      original_release_modified=False, output_directory=str(output))
        totals = Counter()
        with ExitStack() as stack:
            archives = {}
            for batch_records in pending.values():
                free = shutil.disk_usage(scratch_root).free / (1024 ** 3)
                if free < config["minimum_free_gib"]:
                    raise ValueError(f"Scratch space {free:.2f} GiB is below minimum_free_gib")
                raw = raw_archive(Path(config["raw_root"]).resolve(), batch_records)
                with TemporaryDirectory(prefix="box_surface_", dir=scratch_root) as temporary:
                    with closing(raw_episodes(raw, batch_records, Path(temporary))) as episodes:
                        for record, directory in episodes:
                            corners = release_tables(manifest_path, manifest, record, archives, stack)
                            arrays = track_box_episode(directory, corners, tracker, settings)
                            name = f'{record["seed"]}_{record["policy_mode"]}'
                            path = output / "box_surface" / f"{name}.npz"
                            write_episode(path, arrays)
                            added = dict(file=path.relative_to(output).as_posix(), sha256=file_sha256(path),
                                         frames=record["frames"], points=len(arrays["surface_track_id"]),
                                         anchor_frame=int(arrays["surface_anchor_frame"]))
                            receipt = dict(source_record_sha256=canonical_hash(record),
                                           cache_identity_sha256=canonical_hash(identity), box_surface=added)
                            write_json(output / "receipts" / f"{name}.json", receipt)
                            completed[record["seed"], record["policy_mode"]] = dict(deepcopy(record), box_surface=added)
                            totals[record["split"]] += 1
                            print(f"Cached {name} ({record['split']}): {added['points']} Box points", flush=True)
                            logger.log(dict(episodes=len(completed), points=added["points"]), step=len(completed), prefix="cache")
        rows = [completed[row["seed"], row["policy_mode"]] for row in records]
        candidate_path = None
        if config["export_candidate"]:
            from data_collection.five_mode_surface_export import export_candidate

            candidate_path = export_candidate(
                output, manifest_path, manifest, rows, identity, config["episodes_per_shard"],
            )
            logger.link_output(candidate_path, kind="episode_metadata")
        result = dict(
            format=identity["format"], status="COMPLETE" if identity["production_tracking"] else "SMOKE_ONLY",
            run_id=logger.run_id, identity_sha256=canonical_hash(identity),
            base_manifest=dict(path=str(manifest_path), sha256=config["manifest"]["sha256"]),
            condition_index=dict(path=str(index_path), sha256=config["condition_index"]["sha256"]),
            original_member_root=str(manifest_path.parent), counts=counts, records=rows,
            candidate_manifest=None if candidate_path is None else dict(
                path=str(candidate_path), sha256=file_sha256(candidate_path),
            ),
            test_samples_read=False, original_release_modified=False,
        )
        write_json(output / "manifest.json", result)
        logger.link_output(output / "manifest.json", kind="episode_metadata")
        logger.link_output(identity_path, kind="episode_metadata")
        print(f"Completed additive Box surface manifest: {output / 'manifest.json'}", flush=True)
        return output
