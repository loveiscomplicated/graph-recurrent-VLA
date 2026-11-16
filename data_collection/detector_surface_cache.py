"""Build detector-based visibility indices and surface-only TAPNext++ caches."""

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from data_collection.detected_corner_visibility import classify_episode, coplanarity_tolerance
from data_collection.episode_dataset import file_sha256
from data_collection.state21_shared_tapnext import _lift_tracks, split_reset_seeds


def open_sources(config):
    """Pair each complete detector manifest with its exact raw dataset."""
    for settings in config["sources"]:
        raw_path = Path(settings["source_manifest"])
        detector_path = Path(settings["detection_manifest"])
        raw = json.loads(raw_path.read_text())
        detector = json.loads(detector_path.read_text())
        if (
            raw["status"] != "COMPLETE" or detector["status"] != "COMPLETE"
            or detector["camera"] != "base_camera"
            or detector["source_manifest_sha256"] != file_sha256(raw_path)
            or [entry["path"] for entry in raw["episodes"]]
            != [entry["source"] for entry in detector["episodes"]]
        ):
            raise ValueError(f"Detector manifest does not match raw dataset: {detector_path}")
        yield raw_path, detector_path, raw["episodes"], detector["episodes"]


def read_detections(detector_path, receipt, episode, *, include_rim_flags=False):
    """Read checksum-linked per-frame detector points in source order."""
    path = detector_path.parent / receipt["file"]
    if path.name != receipt["file"] or file_sha256(path) != receipt["sha256"]:
        raise ValueError(f"Invalid detector archive: {path}")
    fields = (
        "frame_indices", "corner_offsets", "corner_uv",
        "corner_world_xyz", "corner_object_id",
    )
    if include_rim_flags:
        fields += ("corner_contour_hole",)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in fields}
    count = episode["observations"]
    offsets = arrays["corner_offsets"]
    size = len(arrays["corner_uv"])
    if (
        receipt["frames"] != count or receipt["seed"] != episode["seed"]
        or not np.array_equal(arrays["frame_indices"], np.arange(count))
        or offsets.shape != (count + 1,) or offsets.dtype.kind not in "iu"
        or offsets[0] != 0 or offsets[-1] != size or np.any(np.diff(offsets) < 0)
        or arrays["corner_uv"].shape != (size, 2)
        or arrays["corner_world_xyz"].shape != (size, 3)
        or arrays["corner_object_id"].shape != (size,)
        or not np.isfinite(arrays["corner_uv"]).all()
        or not np.isfinite(arrays["corner_world_xyz"]).all()
        or not np.isin(arrays["corner_object_id"], (0, 1)).all()
    ):
        raise ValueError(f"Invalid detector frame layout: {path}")
    return arrays


def index(config):
    """Index raw paths from per-frame detector corners and the existing seed split."""
    output = Path(config["index_output"])
    if output.exists():
        raise FileExistsError(output)
    tolerance = coplanarity_tolerance(config["coplanarity_tolerance_m"])
    sources = list(open_sources(config))
    split_settings = config["split"]
    splits = split_reset_seeds(
        [episode["seed"] for _, _, episodes, _ in sources for episode in episodes],
        **split_settings,
    )
    split_by_seed = {seed: split for split, seeds in splits.items() for seed in seeds}
    groups = {
        name: {split: [] for split in splits}
        for name in ("condition_1", "condition_2", "condition_3")
    }
    for raw, detector, episodes, receipts in sources:
        for episode_number, (episode, receipt) in enumerate(zip(episodes, receipts, strict=True), 1):
            detections = read_detections(detector, receipt, episode, include_rim_flags=True)
            selected = classify_episode(detections, episode["observations"], tolerance_m=tolerance)
            split = split_by_seed[episode["seed"]]
            directory = raw.parent / episode["path"]
            for condition in groups:
                groups[condition][split].extend(
                    (directory / f"frame_{frame:06d}.npz").as_posix()
                    for frame in selected[condition]
                )
            if episode_number % 50 == 0:
                print(f"{raw.parent.name}: indexed {episode_number}/{len(episodes)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(groups, stream, indent=2)
        stream.write("\n")
    print({condition: {split: len(rows) for split, rows in groups_.items()}
           for condition, groups_ in groups.items()}, flush=True)


def away_from_corners(points, corners, radius):
    """Reject points within the configured image-space radius of a detection."""
    distance_squared = ((points[:, None] - corners[None]) ** 2).sum(axis=-1)
    return (distance_squared > radius ** 2).all(axis=1)


def track_episode(frames, detections, tracker, detector, settings):
    """Query only non-corner Peg surface pixels and emit visible RGB-D points."""
    import torch

    from evaluation.tapnextpp_surface_tracking import choose_points, detector_inputs, track_surface_frame
    from data_collection.shared_tapnext_cache import detector_window_hint

    count = settings["surface_queries"]
    radius = settings["corner_exclusion_px"]
    state = None
    anchor = -1
    initial = np.empty((0, 2), dtype=np.float32)
    offsets = [0]
    surface_uv, world_xyz, track_ids = [], [], []
    for index, frame in enumerate(frames):
        rgb, depth, intrinsics, extrinsics = frame
        rows = slice(*detections["corner_offsets"][index:index + 2])
        corners = detections["corner_uv"][rows]
        hint = detector_window_hint(corners, depth.shape)
        queries = None
        if state is None:
            # The detector's plane fitting requires FP32 even under tracker autocast.
            with torch.autocast("cuda", enabled=False):
                result = detector(*detector_inputs(frame, detector.device, detector.device.type))
            peg_mask = result.masks[0].cpu().numpy().astype(bool)
            candidate_depth = depth.copy()
            y, x = np.nonzero(peg_mask & (depth > 0))
            keep = away_from_corners(np.column_stack((x, y)), corners, radius)
            candidate_depth[y[~keep], x[~keep]] = 0
            try:
                queries = choose_points(peg_mask, candidate_depth, count)
            except ValueError:
                offsets.append(offsets[-1])
                continue
            initial = queries.copy()
            anchor = index
            hint |= peg_mask

        xy, visible, state = track_surface_frame(
            tracker, settings, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            query=queries, state=state, observation_mask=hint,
        )
        if xy.shape != (count, 2) or visible.shape != (count,):
            raise ValueError(f"Unexpected surface track dimensions at frame {index}")
        points = _lift_tracks(
            xy, visible & away_from_corners(xy, corners, radius),
            depth, intrinsics, extrinsics,
        )
        ids = np.asarray(list(points), dtype=np.int32)
        track_ids.append(ids)
        surface_uv.append(xy[ids])
        world_xyz.append(np.asarray(list(points.values()), dtype=np.float32).reshape(-1, 3))
        offsets.append(offsets[-1] + len(ids))

    return {
        **detections,
        "surface_anchor_frame": np.asarray(anchor),
        "surface_query_xy": initial,
        "surface_offsets": np.asarray(offsets, dtype=np.int64),
        "surface_track_id": np.concatenate(track_ids) if track_ids else np.empty(0, dtype=np.int32),
        "surface_uv": np.concatenate(surface_uv) if surface_uv else np.empty((0, 2), dtype=np.float32),
        "surface_world_xyz": np.concatenate(world_xyz) if world_xyz else np.empty((0, 3), dtype=np.float32),
    }


def build(config):
    """Write restartable episode archives and finalize a checksum manifest."""
    import torch
    from data_collection.shared_tapnext_cache import write_archive
    from data_collection.storage import write_json
    from evaluation.tapnextpp_surface_tracking import load_tracker, read_frames
    from visual.corner_detection import TorchCornerDetector

    settings = config["tracking"]
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    tracker = load_tracker(settings)
    detector = TorchCornerDetector.main(device=settings["detector_device"])
    started = perf_counter()
    for raw, detected, episodes, receipts in open_sources(config):
        root = Path(config["cache_root"]) / raw.parent.name
        root.mkdir(parents=True, exist_ok=True)
        identity = {
            "format": "detector_corners_and_visible_tapnext_surface_v1",
            "source_manifest_sha256": file_sha256(raw),
            "detector_manifest_sha256": file_sha256(detected),
            "tracker_checkpoint_sha256": file_sha256(settings["checkpoint"]),
            "builder_sha256": file_sha256(__file__),
            "tracking": settings,
        }
        identity_path = root / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError(f"Output directory belongs to a different build: {root}")
        else:
            write_json(identity_path, identity)
        records = []
        for number, (episode, receipt) in enumerate(zip(episodes, receipts, strict=True), 1):
            name = episode["path"]
            if Path(name).name != name:
                raise ValueError(f"Unexpected episode path: {name}")
            output = root / f"{name}.npz"
            if not output.exists():
                detections = read_detections(detected, receipt, episode)
                frames = read_frames({"source": raw.parent / name}, episode)
                arrays = track_episode(frames, detections, tracker, detector, settings)
                write_archive(output, arrays, verify=True)
            with np.load(output, allow_pickle=False) as saved:
                points = int(saved["surface_offsets"][-1])
                anchor = int(saved["surface_anchor_frame"])
            records.append({"source": name, "file": output.name, "sha256": file_sha256(output),
                            "frames": episode["observations"], "surface_points": points,
                            "surface_anchor_frame": anchor})
            print(f"{root.name} {number}/{len(episodes)} {name} surface={points} "
                  f"elapsed={perf_counter() - started:.1f}s", flush=True)
        write_json(root / "manifest.json", {**identity, "status": "COMPLETE", "episodes": records})
