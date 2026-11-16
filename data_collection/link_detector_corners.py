"""Attach current-frame detector corners to existing TAPNext++ surfaces."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from data_collection.corner_detection_cache import build
from data_collection.episode_dataset import file_sha256


def pack_corners(detector_manifest, shared):
    """Verify episode/frame alignment and pack ragged detector coordinates."""
    manifest = json.loads(detector_manifest.read_text())
    source = json.loads(shared["metadata_json"].item())
    episodes = manifest["episodes"]
    if (
        manifest["status"] != "COMPLETE" or manifest["camera"] != "base_camera"
        or manifest["source_manifest_sha256"] != source["source_manifest_sha256"]
        or [row["source"] for row in episodes] != shared["episode_ids"].tolist()
    ):
        raise ValueError("Detector and tracking caches must describe the same raw episodes")
    keys = ("corner_uv", "corner_world_xyz", "corner_camera_xyz", "corner_object_id",
            "corner_method", "corner_contour_hole")
    values = {key: [] for key in keys}
    counts = []
    for row, first, last in zip(
        episodes, shared["episode_offsets"][:-1], shared["episode_offsets"][1:], strict=True
    ):
        path = detector_manifest.parent / row["file"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Detector cache checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as detector:
            offsets = detector["corner_offsets"]
            if (
                row["frames"] != last - first
                or not np.array_equal(detector["frame_indices"], shared["frame_indices"][first:last])
                or offsets.shape != (last - first + 1,) or offsets.dtype.kind not in "iu"
                or offsets[0] != 0 or offsets[-1] != row["corners"]
                or np.any(np.diff(offsets) < 0)
            ):
                raise ValueError(f"Invalid frame alignment: {path}")
            counts.append(np.diff(offsets))
            for key in keys:
                array = detector[key]
                if len(array) != offsets[-1]:
                    raise ValueError(f"Invalid corner rows for {key}: {path}")
                values[key].append(array)
    packed = {key: np.concatenate(arrays) for key, arrays in values.items()}
    packed["corner_frame_offsets"] = np.r_[0, np.cumsum(np.concatenate(counts), dtype=np.int64)]
    return packed


def visible_surfaces(shared, corners):
    """Extract visible tracks and remove same-pixel overlaps with new corners."""
    xy = shared["surface_xy"]
    keep = shared["surface_visible"].copy()
    corner_pixels = np.rint(corners["corner_uv"]).astype(np.int32)
    offsets = corners["corner_frame_offsets"]
    for frame, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        queries = np.flatnonzero(keep[frame])
        surface_pixels = np.rint(xy[frame, queries]).astype(np.int32)
        overlaps = (surface_pixels[:, None] == corner_pixels[None, start:stop]).all(axis=2)
        keep[frame, queries[overlaps.any(axis=1)]] = False

    frames, ids = np.nonzero(keep)
    return {
        "surface_xy": xy[frames, ids],
        "surface_track_id": ids.astype(np.int32),
        "surface_frame_offsets": np.r_[0, np.cumsum(keep.sum(axis=1), dtype=np.int64)],
    }


def save_archive(path, fields, metadata):
    """Verify the serialized arrays before creating a new output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent) as stream:
        arrays = {**fields, "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}
        np.savez_compressed(stream, **arrays)
        stream.flush()
        with np.load(stream.name, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        path.hardlink_to(stream.name)


def link(settings, detector_manifest):
    """Create independent dense and visible-surface variants with the same detector rows."""
    source = Path(settings["tracking_cache"])
    with np.load(source, allow_pickle=False) as cache:
        corners = pack_corners(detector_manifest, cache)
        shared = {key: cache[key] for key in cache.files
                  if key != "metadata_json" and not key.startswith("corner_")}
        provenance = json.loads(cache["metadata_json"].item())
    metadata = {
        "format_version": 1,
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "surface_source": str(source),
        "surface_source_sha256": file_sha256(source),
        "detector_manifest": str(detector_manifest),
        "detector_manifest_sha256": file_sha256(detector_manifest),
        "corner_ids": "frame-local detector order; object IDs 0=Peg, 1=Box",
        "corner_coordinates": "original image xy in pixels and world/camera XYZ in metres",
        "surface_ids": "query index within the episode",
        "surface_coordinates": "original image xy in pixels",
        "original_tracking_metadata": provenance,
    }
    dense_output = Path(settings["linked_tracking_cache"])
    visible_output = Path(settings["linked_visible_cache"])
    frame_keys = ("episode_ids", "episode_offsets", "frame_indices")
    surface_fields = visible_surfaces(shared, corners)
    save_archive(dense_output, {**shared, **corners}, {
        **metadata, "format": "tapnext_state36_with_detector_corners",
    })
    save_archive(visible_output, {
        **{key: shared[key] for key in frame_keys}, **surface_fields, **corners,
    }, {
        **metadata, "format": "tapnext_visible_surface_with_detector_corners",
        "surface_filter": "surface_visible and no same-frame rounded-pixel detector corner overlap",
    })
    print(f"{detector_manifest}: {len(corners['corner_uv']):,} corners", flush=True)
    print(f"{dense_output}\n{visible_output}: {len(surface_fields['surface_xy']):,} visible surface points",
          flush=True)


def run(config):
    """Generate detections or reuse a completed detector manifest, then link it."""
    for key in ("linked_tracking_cache", "linked_visible_cache"):
        path = Path(config[key])
        if path.exists():
            raise FileExistsError(path)
    manifest = config.get("detection_manifest")
    if manifest is None:
        manifest = build(config["detector"])
    link(config, Path(manifest))
