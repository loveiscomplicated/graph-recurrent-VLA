"""Export typed Peg/Box surfaces while preserving original detector and target bytes."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import tarfile

import numpy as np

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.five_mode_surface_source import ALLOWED_SPLITS, BASE_KINDS, release_member_bytes
from data_collection.storage import write_json


def merge_surfaces(peg, box):
    """Append Box rows after Peg rows in every frame and preserve the Peg projection exactly."""
    if not np.array_equal(peg["frame_indices"], box["frame_indices"]):
        raise ValueError("Box surface frame IDs differ from the original Peg cache")
    frames = len(peg["frame_indices"])
    for table, object_id in ((peg, 0), (box, 1)):
        offsets = table["surface_offsets"]
        count = len(table["surface_track_id"])
        query_count = len(table["surface_query_xy"])
        if (offsets.shape != (frames + 1,) or offsets.dtype.kind not in "iu"
                or offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) < 0)
                or table["surface_uv"].shape != (count, 2)
                or table["surface_world_xyz"].shape != (count, 3)
                or table["surface_query_xy"].shape != (query_count, 2)
                or table["surface_track_id"].dtype.kind not in "iu"
                or np.any(table["surface_track_id"] < 0) or np.any(table["surface_track_id"] >= query_count)
                or not all(np.isfinite(table[name]).all() for name in
                           ("surface_uv", "surface_world_xyz", "surface_query_xy"))):
            raise ValueError("Invalid surface offsets, coordinates or track/query identities")
        if "surface_object_id" in table:
            kinds = table["surface_object_id"]
            if kinds.shape != (count,) or kinds.dtype.kind not in "iu" or np.any(kinds != object_id):
                raise ValueError("Expected original Peg-only surface and an added Box-only surface")
    query_offset = len(peg["surface_query_xy"])
    indices = []
    for frame in range(frames):
        for object_id, table in enumerate((peg, box)):
            start, stop = table["surface_offsets"][frame:frame + 2]
            indices.extend((object_id, row) for row in range(start, stop))
    result = {name: value.copy() for name, value in peg.items()}
    object_ids = np.asarray([object_id for object_id, _ in indices], dtype=np.int8)
    for name, width in (("surface_uv", 2), ("surface_world_xyz", 3)):
        if peg[name].dtype != box[name].dtype:
            raise ValueError(f"Adding Box points would change the original Peg dtype: {name}")
        result[name] = np.asarray(
            [(peg, box)[object_id][name][row] for object_id, row in indices], dtype=peg[name].dtype,
        ).reshape(-1, width)
    result["surface_track_id"] = np.asarray([
        (peg, box)[object_id]["surface_track_id"][row] + (query_offset if object_id else 0)
        for object_id, row in indices
    ], dtype=np.result_type(peg["surface_track_id"].dtype, box["surface_track_id"].dtype))
    counts = np.diff(peg["surface_offsets"]) + np.diff(box["surface_offsets"])
    result.update(
        surface_offsets=np.r_[0, counts.cumsum()].astype(np.int64), surface_object_id=object_ids,
        surface_query_xy=np.concatenate((peg["surface_query_xy"], box["surface_query_xy"])),
        surface_query_object_id=np.r_[np.zeros(query_offset, dtype=np.int8),
                                      np.ones(len(box["surface_query_xy"]), dtype=np.int8)],
        surface_object_anchor_frames=np.asarray([peg["surface_anchor_frame"], box["surface_anchor_frame"]]),
    )
    anchors = result["surface_object_anchor_frames"]
    result["surface_anchor_frame"] = np.asarray(min(anchors[anchors >= 0], default=-1), dtype=np.int64)
    # Reconstruct the Peg-only ragged layout, not just its unordered point set.
    keep = object_ids == 0
    frame_ids = np.repeat(np.arange(frames), counts)
    restored_offsets = np.r_[0, np.bincount(frame_ids[keep], minlength=frames).cumsum()]
    if not np.array_equal(restored_offsets, peg["surface_offsets"]):
        raise ValueError("Candidate surface changes Peg frame offsets")
    for name in ("surface_uv", "surface_world_xyz", "surface_track_id"):
        if result[name][keep].dtype != peg[name].dtype or not np.array_equal(result[name][keep], peg[name]):
            raise ValueError(f"Candidate surface changes original Peg values/order/dtype: {name}")
    return result


def export_candidate(output, manifest_path, manifest, rows, identity, episodes_per_shard):
    """Write a standalone train/validation release accepted by the explicit Box-input loader."""
    if any(row["split"] not in ALLOWED_SPLITS for row in rows):
        raise ValueError("Candidate export cannot include test episodes")
    candidate = output / "candidate"
    candidate.mkdir(exist_ok=True)
    identity_hash = canonical_hash(identity)
    source_hash = canonical_hash(rows)
    identity_path = candidate / "identity.json"
    expected = dict(augmentation_identity_sha256=identity_hash, source_records_sha256=source_hash)
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != expected:
            raise ValueError("Candidate export directory belongs to a different augmentation")
    else:
        if any(candidate.iterdir()):
            raise ValueError("Candidate export needs an empty directory or its matching identity")
        write_json(identity_path, expected)
    result_path = candidate / "manifest.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if any(previous[key] != value for key, value in expected.items()):
            raise ValueError("Candidate manifest identity differs")
        for receipt in previous["archives"]:
            path = (candidate / receipt["file"]).resolve()
            if (not path.is_relative_to(candidate) or path.suffix != ".tar"
                    or file_sha256(path) != receipt["sha256"]):
                raise ValueError("Candidate shard checksum differs")
        return result_path
    exported, shards = [], []
    with ExitStack() as stack:
        archives = {}
        for start in range(0, len(rows), episodes_per_shard):
            filename = f"shard_{len(shards):03d}.tar"
            path = candidate / filename
            temporary = path.with_suffix(".partial")
            with tarfile.open(temporary, "w") as target:
                for row in rows[start:start + episodes_per_shard]:
                    updated = deepcopy(row)
                    updated["files"] = {}
                    updated["box_surface"]["root"] = str(output)
                    for kind in BASE_KINDS:
                        receipt = row["files"][kind]
                        contents = release_member_bytes(manifest_path, manifest, receipt, kind, archives, stack)
                        if kind == "surface":
                            with np.load(BytesIO(contents), allow_pickle=False) as arrays:
                                peg = dict(arrays)
                            added = row["box_surface"]
                            box_path = output / added["file"]
                            if file_sha256(box_path) != added["sha256"]:
                                raise ValueError("Added Box surface checksum differs during export")
                            with np.load(box_path, allow_pickle=False) as arrays:
                                merged = merge_surfaces(peg, dict(arrays))
                            buffer = BytesIO()
                            np.savez_compressed(buffer, **merged)
                            contents = buffer.getvalue()
                            updated["original_peg_surface_sha256"] = receipt["sha256"]
                        member = f'{row["batch"]}/{kind}/{row["seed"]}_{row["policy_mode"]}.npz'
                        info = tarfile.TarInfo(member)
                        info.size = len(contents)
                        target.addfile(info, BytesIO(contents))
                        updated["files"][kind] = dict(
                            archive=filename, member=member, sha256=hashlib.sha256(contents).hexdigest(),
                        )
                    exported.append(updated)
            temporary.replace(path)
            shards.append(dict(file=filename, sha256=file_sha256(path), bytes=path.stat().st_size))
    production = identity["production_tracking"]
    result = dict(
        format="five_mode_supervised_v1" if production else "five_mode_box_surface_smoke_v1",
        status="COMPLETE" if production else "SMOKE_ONLY", **expected,
        seeds=len({row["seed"] for row in rows}), episodes=len(rows), frames=sum(row["frames"] for row in rows),
        episodes_by_split={split: sum(row["split"] == split for row in rows) for split in ALLOWED_SPLITS},
        base_manifest=dict(path=str(manifest_path), sha256=identity["manifest_sha256"]),
        condition_index_sha256=identity["condition_index_sha256"], target_fields=manifest.get("target_fields"),
        records=exported, archives=shards, test_samples_read=False,
    )
    write_json(result_path, result)
    return result_path
