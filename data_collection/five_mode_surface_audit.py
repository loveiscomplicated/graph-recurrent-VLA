"""Inspect object-identified surface availability in approved Five-mode shards."""

from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import tarfile

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from training.gnn.packed_five_mode_data import indexed_frames, read_archive_member
from utils.experiment_logger import ExperimentLogger


def pinned_json(settings):
    """Read an explicitly selected manifest/index after verifying its exact bytes."""
    path = Path(settings["path"]).resolve()
    if file_sha256(path) != settings["sha256"]:
        raise ValueError(f"Five-mode input checksum mismatch: {path}")
    return path, json.loads(path.read_text())


def surface_counts(surface, frame_count):
    """Count typed points without guessing the object identity of unlabeled coordinates."""
    offsets = surface["surface_offsets"]
    xyz = surface["surface_world_xyz"]
    if (not np.array_equal(surface["frame_indices"], np.arange(frame_count))
            or offsets.shape != (frame_count + 1,) or offsets.dtype.kind not in "iu"
            or offsets[0] != 0 or offsets[-1] != len(xyz) or np.any(np.diff(offsets) < 0)
            or xyz.shape != (len(xyz), 3) or not np.isfinite(xyz).all()):
        raise ValueError("Invalid frame-aligned Five-mode surface arrays")
    ids = surface.get("surface_object_id")
    counts = {name: np.zeros(frame_count, dtype=np.int64) for name in ("peg", "box", "untyped")}
    if ids is None:
        counts["untyped"] = np.diff(offsets)
    else:
        if ids.shape != (len(xyz),) or ids.dtype.kind not in "iu" or not np.isin(ids, (0, 1)).all():
            raise ValueError("surface_object_id must identify every point as Peg=0 or Box=1")
        frames = np.repeat(np.arange(frame_count), np.diff(offsets))
        for object_id, name in enumerate(("peg", "box")):
            counts[name] = np.bincount(frames[ids == object_id], minlength=frame_count)
    return counts, ids is not None


def audit(config):
    """Read every train/validation surface member; never open test or target members."""
    if config.get("splits") != ["train", "validation"]:
        raise ValueError("Surface audit requires splits: [train, validation]; test is reserved")
    manifest_path, manifest = pinned_json(config["manifest"])
    _, index = pinned_json(config["condition_index"])
    records = manifest["records"]
    assignments = {(row["seed"], row["policy_mode"]): row["split"] for row in records}
    if len(assignments) != len(records):
        raise ValueError("Five-mode manifest contains duplicate episodes")
    selected = indexed_frames(index, "condition_2", ("train", "validation"), assignments)
    archive_names = {row["file"] for row in manifest["archives"]}
    result = {split: {"all": Counter(), "condition_2": Counter()} for split in config["splits"]}
    receipts = []
    with ExperimentLogger(team="graph", model="surface_inventory", config=config,
                          job_type="evaluate") as logger:
        with ExitStack() as stack:
            archives = {}
            for episode in records:
                split = episode["split"]
                if split not in result:
                    continue
                receipt = episode["files"]["surface"]
                name = receipt["archive"]
                path = (manifest_path.parent / name).resolve()
                if name not in archive_names or not path.is_relative_to(manifest_path.parent):
                    raise ValueError(f"Unexpected supervised surface archive: {path}")
                if path not in archives:
                    archives[path] = stack.enter_context(tarfile.open(path, "r:"))
                surface = read_archive_member(archives[path], receipt, "surface")
                counts, typed = surface_counts(surface, episode["frames"])
                key = episode["seed"], episode["policy_mode"]
                for cohort, frames in (("all", np.arange(episode["frames"])),
                                       ("condition_2", np.asarray(selected[key], dtype=np.int64))):
                    summary = result[split][cohort]
                    summary["episodes"] += int(len(frames) > 0)
                    summary["frames"] += len(frames)
                    summary["episodes_with_object_ids"] += int(typed and len(frames) > 0)
                    for object_name, per_frame in counts.items():
                        summary[f"{object_name}_points"] += int(per_frame[frames].sum())
                        summary[f"frames_with_{object_name}"] += int(np.count_nonzero(per_frame[frames]))
                receipts.append(dict(seed=key[0], policy_mode=key[1], split=split, **receipt))
        for split, expected in config["condition_2_frames"].items():
            if result[split]["condition_2"]["frames"] != expected:
                raise ValueError(f"Unexpected audited C2/{split} frame count")
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        box_points = sum(row["all"]["box_points"] for row in result.values())
        report = dict(
            run_id=logger.run_id, config=config, counts=result,
            identified_box_surface_available=box_points > 0,
            untyped_interpretation="Producer's no-plan path is Peg-only; coordinates alone do not establish Box identity",
            test_samples_read=False, target_members_read=False,
        )
        write_json(output / "results.json", report)
        write_json(output / "verified_surface_members.json", receipts)
        for filename in ("results.json", "verified_surface_members.json"):
            logger.link_output(output / filename, kind="metrics")
        logger.record(
            manifest_sha256=config["manifest"]["sha256"],
            condition_index_sha256=config["condition_index"]["sha256"],
            auditor_sha256=file_sha256(Path(__file__)),
            test_samples_read=False, target_members_read=False,
        )
        logger.log(result, step=1, prefix="surface_inventory")
        print(f"Audited {len(receipts):,} surface members; identified Box points={box_points}: {output}", flush=True)
        return output
