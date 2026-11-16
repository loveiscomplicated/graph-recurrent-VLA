"""Pack a shared TAPNext++ observation and State36 cache for recorded episodes."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.tapnextpp_surface_tracking import (
    choose_points,
    detector_inputs,
    load_tracker,
    read_frames,
    track_surface_frame,
)
from visual.corner_detection import TorchCornerDetector


def valid_corners(uv, image_shape):
    """Identify detector queries inside the source image."""
    height, width = image_shape
    return (
        np.isfinite(uv).all(axis=1)
        & (uv >= 0).all(axis=1)
        & (uv < [width, height]).all(axis=1)
    )


def detector_window_hint(uv, image_shape):
    """Use the cached detection bounds as an optional adaptive-window hint."""
    mask = np.zeros(image_shape, dtype=bool)
    pixels = uv[valid_corners(uv, image_shape)].astype(np.int64)
    mask[pixels[:, 1], pixels[:, 0]] = True
    return mask


def observe_episode(episode, frames, detections, tracker, detector, settings):
    """Track initial Peg surface and detector corner queries in one causal stream."""
    count = len(frames)
    surface_count = settings["surface_queries"]
    surface_xy = np.full((count, surface_count, 2), np.nan, dtype=np.float32)
    surface_visible = np.zeros((count, surface_count), dtype=bool)
    surface_queries = surface_xy[0].copy()
    corner_queries = np.empty((0, 2), dtype=np.float32)
    corner_kinds = np.empty(0, dtype=np.int8)
    corner_xy = np.empty((count, 0, 2), dtype=np.float32)
    corner_visible = np.zeros((count, 0), dtype=bool)
    anchor = -1
    state = None
    for index, frame in enumerate(frames):
        rgb, depth, _, _ = frame
        indices = slice(*detections["corner_offsets"][index : index + 2])
        observed_uv = detections["corner_uv"][indices]
        hint = detector_window_hint(observed_uv, depth.shape)
        queries = None
        if state is None:
            # Surface queries use the original depth-supported Peg-mask sampler.
            with torch.autocast("cuda", enabled=False):
                first = detector(*detector_inputs(frame, detector.device, "cuda"))
            peg_mask = first.masks[0].cpu().numpy().astype(bool)
            try:
                surface_queries = choose_points(peg_mask, depth, surface_count)
            except ValueError:
                continue
            keep = valid_corners(observed_uv, depth.shape)
            corner_queries = observed_uv[keep]
            corner_kinds = detections["corner_object_id"][indices][keep]
            queries = np.concatenate((surface_queries, corner_queries))
            corner_xy = np.full(
                (count, len(corner_queries), 2), np.nan, dtype=np.float32
            )
            corner_visible = np.zeros((count, len(corner_queries)), dtype=bool)
            anchor = index
            hint |= peg_mask

        xy, visible, state = track_surface_frame(
            tracker,
            settings,
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            query=queries,
            state=state,
            observation_mask=hint,
        )
        if (
            xy.shape != (surface_count + len(corner_queries), 2)
            or visible.shape != xy.shape[:1]
        ):
            raise ValueError(f"Unexpected query count in {episode} frame {index}")
        if not np.isfinite(xy[visible]).all():
            raise ValueError(
                f"Nonfinite visible TAPNext++ output in {episode} frame {index}"
            )
        surface_xy[index], surface_visible[index] = (
            xy[:surface_count],
            visible[:surface_count],
        )
        corner_xy[index], corner_visible[index] = (
            xy[surface_count:],
            visible[surface_count:],
        )

    return {
        "surface_anchor_frame": np.asarray(anchor),
        "surface_query_xy": surface_queries,
        "surface_xy": surface_xy,
        "surface_visible": surface_visible,
        "corner_query_xy": corner_queries,
        "corner_query_object_id": corner_kinds,
        "corner_xy": corner_xy,
        "corner_visible": corner_visible,
    }


def read_detection_archive(path, record, frames):
    """Verify the source archive and its ragged per-frame coordinates."""
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"Corner detection archive SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {
            name: saved[name]
            for name in (
                "frame_indices",
                "corner_offsets",
                "corner_uv",
                "corner_object_id",
            )
        }
    if not np.array_equal(arrays["frame_indices"], np.arange(frames)):
        raise ValueError(f"Wrong frame indices in detector cache: {path}")
    offsets = arrays["corner_offsets"]
    row_count = len(arrays["corner_uv"])
    if (
        offsets.shape != (frames + 1,)
        or offsets[0] != 0
        or offsets[-1] != row_count
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError(f"Invalid detector offsets in {path}")
    if arrays["corner_uv"].shape != (row_count, 2) or arrays[
        "corner_object_id"
    ].shape != (row_count,):
        raise ValueError(f"Invalid detector node arrays in {path}")
    return arrays


def write_archive(path, arrays, *, verify=False):
    """Place only complete archives at the final path."""
    temporary = path.with_suffix(".partial")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        if verify:
            with np.load(temporary, allow_pickle=False) as saved:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(value, saved[key])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def pack(ground_truth, work, identities, metadata, output):
    """Pack each track once, with frame and episode offsets for ragged corners."""
    episode_offsets = ground_truth["episode_offsets"]
    arrays = {
        "surface_anchor_frame": [],
        "surface_query_xy": [],
        "surface_xy": [],
        "surface_visible": [],
        "corner_query_xy": [],
        "corner_query_object_id": [],
        "corner_xy": [],
        "corner_visible": [],
    }
    queries_per_episode, queries_per_frame = [], []
    for index, identity in enumerate(identities):
        with np.load(work / f"{identity}.npz", allow_pickle=False) as saved:
            frames = episode_offsets[index + 1] - episode_offsets[index]
            queries = len(saved["corner_query_xy"])
            if saved["surface_xy"].shape != (
                frames,
                metadata["tracking_settings"]["surface_queries"],
                2,
            ) or saved["corner_xy"].shape != (frames, queries, 2):
                raise ValueError(f"Unexpected tracking dimensions for {identity}")
            queries_per_episode.append(queries)
            queries_per_frame.extend([queries] * frames)
            for name, rows in arrays.items():
                value = saved[name]
                if name == "corner_xy":
                    value = value.reshape(-1, 2)
                elif name == "corner_visible":
                    value = value.ravel()
                rows.append(value)

    stacked = {"surface_anchor_frame", "surface_query_xy"}
    tracks = {
        name: (np.stack(rows) if name in stacked else np.concatenate(rows))
        for name, rows in arrays.items()
    }
    tracks["corner_query_offsets"] = np.r_[0, np.cumsum(queries_per_episode)]
    tracks["corner_frame_offsets"] = np.r_[0, np.cumsum(queries_per_frame)]
    output_arrays = {
        **ground_truth,
        **tracks,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    write_archive(output, output_arrays, verify=True)
    return file_sha256(output)


def build(settings):
    """Run the selected full dataset with checksum-verified detector and GT inputs."""
    settings = {
        "tap_input_resolution": 256,
        "tap_cuda_autocast": False,
        "tap_cache_cuda_autocast": False,
        "tap_mps_fp16_vit": False,
        "tap_mps_fp16_recurrent_mlp": False,
        "tap_mps_skip_past_query_embeddings": False,
        "tap_mps_compile_recurrent": False,
        **settings,
    }
    manifest = Path(settings["source_manifest"])
    detector_manifest = Path(settings["detection_manifest"])
    targets = Path(settings["state36_ground_truth"])
    source = json.loads(manifest.read_text())
    detection = json.loads(detector_manifest.read_text())
    manifest_hash = file_sha256(manifest)
    if (
        source["status"] != "COMPLETE"
        or detection["status"] != "COMPLETE"
        or detection["source_manifest_sha256"] != manifest_hash
    ):
        raise ValueError(
            "Expected complete raw and detector manifests from the same source"
        )

    with np.load(targets, allow_pickle=False) as saved:
        gt = {key: saved[key] for key in saved.files}
    gt_metadata = json.loads(gt.pop("metadata_json").item())
    episode_ids = gt["episode_ids"].tolist()
    states = gt["state36_gt"]
    if (
        states.shape != (len(gt["frame_indices"]), 36)
        or states.dtype != np.float32
        or not np.isfinite(states).all()
    ):
        raise ValueError("State36 GT must contain finite float32[N,36] rows")
    if gt_metadata["source_manifest_sha256"] != manifest_hash:
        raise ValueError("State36 targets and source manifest differ")
    if not (
        episode_ids
        == [row["path"] for row in source["episodes"]]
        == [row["source"] for row in detection["episodes"]]
    ):
        raise ValueError("Raw, GT and detector episode order must match")

    output = Path(settings["output"])
    work = output.parent / "episodes"
    work.mkdir(parents=True, exist_ok=True)
    identity_file = work / "identity.json"
    metadata = {
        **gt_metadata,
        "format": "shared_tapnext_state36",
        "state36_ground_truth_sha256": file_sha256(targets),
        "detector_manifest_sha256": file_sha256(detector_manifest),
        "tapnext_checkpoint_sha256": file_sha256(settings["checkpoint"]),
        "cache_builder_sha256": file_sha256(__file__),
        "tracking_settings": {
            key: value
            for key, value in settings.items()
            if key.startswith(("device", "tap_", "surface_"))
        },
        "observation_coordinates": "original image xy in pixels",
        "corner_ids": "stable initial detector-query index within an episode",
        "corner_world_geometry": "derive from source RGB-D where depth supports the track",
        "surface_ids": "stable initial Peg surface query index within an episode",
        "adaptive_window_hint": "bounds of frame-wise cached detector UV with continued tracks",
    }
    if identity_file.exists() and json.loads(identity_file.read_text()) != metadata:
        raise ValueError(
            f"Existing episode work has different source or tracking settings: {work}"
        )
    if not identity_file.exists():
        identity_file.write_text(json.dumps(metadata, sort_keys=True))
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    tracker = load_tracker(settings)
    detector = TorchCornerDetector.main(device=settings["device"])
    started = perf_counter()
    for index, (episode, record) in enumerate(
        zip(source["episodes"], detection["episodes"], strict=True)
    ):
        name = episode_ids[index]
        if Path(name).name != name or Path(record["file"]).name != record["file"]:
            raise ValueError(
                "Expected episode and detector file names within the dataset"
            )
        begin, end = gt["episode_offsets"][index : index + 2]
        count = end - begin
        if not (
            count == episode["observations"] == record["frames"]
            and episode["seed"] == record["seed"] == gt["reset_seeds"][index]
            and np.array_equal(gt["frame_indices"][begin:end], np.arange(count))
        ):
            raise ValueError(
                f"Source, detector and State36 frame alignment differs for {name}"
            )
        path = work / f"{name}.npz"
        if not path.is_file():
            observed = read_frames({"source": manifest.parent / name}, episode)
            detections = read_detection_archive(
                detector_manifest.parent / record["file"], record, count
            )
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=settings["tap_cache_cuda_autocast"]
            ):
                arrays = observe_episode(
                    name, observed, detections, tracker, detector, settings
                )
            write_archive(path, arrays)
        print(
            f"{index + 1}/{len(episode_ids)} {name} frames={end} elapsed={perf_counter() - started:.1f}s",
            flush=True,
        )
    print(f"{output}: {pack(gt, work, episode_ids, metadata, output)}", flush=True)


def main():
    """Read a JSON run configuration and build one cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    build(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
