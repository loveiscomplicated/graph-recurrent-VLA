"""Read per-frame detector coordinates for selected raw recordings."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256


def read_direct_samples(settings, selected, camera):
    """Yield indexed samples from the detector cache with verified raw targets."""
    manifest_path = Path(settings["source_manifest"]).resolve()
    detection_path = Path(settings["detection_manifest"]).resolve()
    source = json.loads(manifest_path.read_text())
    detection = json.loads(detection_path.read_text())
    source_hash = file_sha256(manifest_path)
    if (source["status"] != "COMPLETE" or detection["status"] != "COMPLETE"
            or detection["source_manifest_sha256"] != source_hash
            or detection["camera"] != camera):
        raise ValueError(f"Detector cache does not match the raw source and camera: {detection_path}")
    records = {row["source"]: row for row in detection["episodes"]}
    if len(records) != len(detection["episodes"]):
        raise ValueError(f"Duplicate detector episodes: {detection_path}")

    identity = {
        "manifest_sha256": source_hash,
        "detection_manifest_sha256": file_sha256(detection_path),
    }
    for episode in source["episodes"]:
        directory = manifest_path.parent / episode["path"]
        hashes = episode["frame_sha256"]
        frames = [(index, directory / f"frame_{index:06d}.npz") for index in range(len(hashes))]
        frames = [(index, path) for index, path in frames if path in selected]
        if not frames:
            continue
        record = records[episode["path"]]
        archive = detection_path.parent / record["file"]
        count = len(hashes)
        if record["frames"] != count or file_sha256(archive) != record["sha256"]:
            raise ValueError(f"Detector archive checksum or count mismatch: {archive}")
        with np.load(archive, allow_pickle=False) as saved:
            offsets = saved["corner_offsets"]
            xyz = saved["corner_world_xyz"]
            kinds = saved["corner_object_id"]
            if (not np.array_equal(saved["frame_indices"], np.arange(count))
                    or offsets.shape != (count + 1,) or offsets.dtype.kind not in "iu"
                    or offsets[0] != 0 or offsets[-1] != len(xyz)
                    or np.any(offsets[1:] < offsets[:-1])
                    or xyz.shape != (len(kinds), 3) or kinds.ndim != 1
                    or not np.isfinite(xyz).all() or not np.isin(kinds, (0, 1)).all()):
                raise ValueError(f"Invalid frame offsets or corner arrays: {archive}")
        for index, path in frames:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != hashes[index]:
                raise ValueError(f"Raw frame checksum mismatch: {path}")
            with np.load(io.BytesIO(content), allow_pickle=False) as raw:
                gt = {
                    field: raw[f"ground_truth/{field}"]
                    for field in ("ee_pose", "peg_pose", "hole_pose", "peg_half_sizes", "hole_radius")
                }
            node_slice = slice(offsets[index], offsets[index + 1])
            yield selected.pop(path), path, xyz[node_slice], kinds[node_slice], gt, identity
