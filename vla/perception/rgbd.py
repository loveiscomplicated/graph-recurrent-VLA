"""Join Visual tracker masks with their recorded RGB-D observations.

Only camera observations and TCP kinematics are read. Object GT, simulator
segmentation IDs, point sampling, and graph construction are outside this adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from operator import index
from pathlib import Path

import numpy as np

# Visual output object IDs consumed by Graph, distinct from graph node types.
PEG = 0
BOX = 1
TRACKER_TO_GRAPH_ID = {1: PEG, 2: BOX}
MASK_NAME_TO_TRACKER_ID = {"peg": 1, "box": 2}


@dataclass(frozen=True)
class TrackedRGBDFrame:
    """One source observation with masks keyed by Visual output object ID.

    Arrays are CPU NumPy arrays. Depth is recorded CV-camera z in metres;
    ``cam2world`` maps CV-camera coordinates to world. ``tcp_pose`` is the
    world pose ``[x, y, z, qw, qx, qy, qz]``. ``source_frame`` is the episode
    array row, while ``frame_idx`` is the recorded simulator step, if present.
    Missing tracked objects have empty boolean masks. Arrays may share storage
    with the input; consumers should not modify them in place.
    """

    source_frame: int
    camera: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    cam2world: np.ndarray
    tcp_pose: np.ndarray
    masks: dict[int, np.ndarray]
    frame_idx: int | None
    timestamp: float | None


def _observations(episode: Mapping[str, object], camera: str) -> dict[str, np.ndarray]:
    keys = {
        name: f"{camera}/{name}" for name in ("rgb", "depth", "intrinsic", "cam2world")
    }
    keys["tcp_pose"] = "tcp_pose"
    missing = [key for key in keys.values() if key not in episode]
    if missing:
        raise ValueError(
            f"episode is missing required observations: {', '.join(missing)}"
        )
    arrays = {name: np.asarray(episode[key]) for name, key in keys.items()}
    rgb = arrays["rgb"]
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"{camera}/rgb must have shape (T,H,W,3), got {rgb.shape}")
    count, height, width = rgb.shape[:3]
    shapes = {
        "depth": (count, height, width),
        "intrinsic": (count, 3, 3),
        "cam2world": (count, 4, 4),
        "tcp_pose": (count, 7),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"{keys[name]} must have shape {shape}, got {arrays[name].shape}"
            )
    if not np.issubdtype(arrays["depth"].dtype, np.floating):
        raise ValueError(
            "recorded depth must be floating point metres, not raw millimetres"
        )
    for name in ("frame_idx", "timestamp"):
        if name in episode:
            arrays[name] = np.asarray(episode[name])
            if arrays[name].shape != (count,):
                raise ValueError(
                    f"{name} must have shape ({count},), got {arrays[name].shape}"
                )
    return arrays


def _source_frames(values: Sequence[int] | np.ndarray, count: int) -> np.ndarray:
    frames = np.asarray(values)
    if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("source_frame must be a one-dimensional integer array")
    if np.any(frames < 0) or np.any(frames >= count):
        raise ValueError(f"source_frame indices must be in [0, {count})")
    return frames


def attach_rgbd(
    tracking: Iterable[tuple[int, Mapping[int, object]]],
    episode: Mapping[str, object],
    *,
    camera: str = "base_camera",
    source_frames: Sequence[int] | np.ndarray | None = None,
) -> Iterator[TrackedRGBDFrame]:
    """Attach recorded observations to the existing ``SAM2Tracker.track`` output.

    Tracker indices address its processed video. If that video was subsampled,
    pass the exact source row for every processed frame in ``source_frames``.
    Direct full-episode tracking uses the tracker index as the source row.
    Torch masks are detached and copied to CPU without importing SAM2 or Torch.
    Unknown tracker IDs and mismatched image sizes raise instead of being guessed
    or resized. Empty masks and invalid recorded depth values are preserved.
    """
    arrays = _observations(episode, camera)
    count, height, width = arrays["rgb"].shape[:3]
    selected = None if source_frames is None else _source_frames(source_frames, count)
    tracker_count = count if selected is None else len(selected)
    for local_frame, tracker_masks in tracking:
        try:
            local_frame = index(local_frame)
        except TypeError as exc:
            raise ValueError("tracker frame index must be an integer") from exc
        if not 0 <= local_frame < tracker_count:
            raise ValueError(
                f"tracker frame index {local_frame} must be in [0, {tracker_count})"
            )
        source = local_frame if selected is None else int(selected[local_frame])
        masks = {
            graph_id: np.zeros((height, width), dtype=bool)
            for graph_id in TRACKER_TO_GRAPH_ID.values()
        }
        for tracker_id, mask in tracker_masks.items():
            if tracker_id not in TRACKER_TO_GRAPH_ID:
                raise ValueError(
                    f"unknown tracker object ID {tracker_id}; expected 1 (peg) or 2 (box)"
                )
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != (height, width):
                raise ValueError(
                    f"tracker mask {tracker_id} must be boolean with shape "
                    f"{(height, width)}, got {mask.dtype} {mask.shape}"
                )
            masks[TRACKER_TO_GRAPH_ID[tracker_id]] = mask
        yield TrackedRGBDFrame(
            source_frame=source,
            camera=camera,
            rgb=arrays["rgb"][source],
            depth=arrays["depth"][source],
            intrinsic=arrays["intrinsic"][source],
            cam2world=arrays["cam2world"][source],
            tcp_pose=arrays["tcp_pose"][source],
            masks=masks,
            frame_idx=(
                int(arrays["frame_idx"][source]) if "frame_idx" in arrays else None
            ),
            timestamp=(
                float(arrays["timestamp"][source]) if "timestamp" in arrays else None
            ),
        )


def iter_saved_rgbd(
    episode_path: str | Path,
    masks_path: str | Path,
    *,
    camera: str = "base_camera",
) -> Iterator[TrackedRGBDFrame]:
    """Join runtime ``masks.npz`` with its original episode using ``source_frame``.

    Pass the original episode and camera used by ``run_tracking.py`` (recorded
    in its ``tracking.json``). Frame rows are never inferred from array lengths
    or timestamps. No detector, tracker checkpoint, or simulator is required.
    """
    with np.load(episode_path, allow_pickle=False) as episode, np.load(
        masks_path, allow_pickle=False
    ) as saved:
        required = ["source_frame", *MASK_NAME_TO_TRACKER_ID]
        missing = [name for name in required if name not in saved]
        if missing:
            raise ValueError(
                f"mask archive is missing required arrays: {', '.join(missing)}"
            )
        source_frames = saved["source_frame"]
        if source_frames.ndim != 1:
            raise ValueError("source_frame must be a one-dimensional integer array")
        masks = {
            tracker_id: saved[name]
            for name, tracker_id in MASK_NAME_TO_TRACKER_ID.items()
        }
        for tracker_id, values in masks.items():
            if values.ndim != 3 or len(values) != len(source_frames):
                raise ValueError(
                    f"saved masks for tracker {tracker_id} must have shape "
                    "(len(source_frame), H, W)"
                )
        tracking = (
            (row, {tracker_id: values[row] for tracker_id, values in masks.items()})
            for row in range(len(source_frames))
        )
        yield from attach_rgbd(
            tracking, episode, camera=camera, source_frames=source_frames
        )
