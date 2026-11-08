"""Select physical vertex coverage from raw detections or cached graph points."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from data_collection.episode_dataset import file_sha256, read_frame
from data_collection.physical_vertex_filter import peg_corner_ids, physical_vertex_conditions
from data_collection.storage import write_json
from data_collection.vertex_family_labels import CertificationSettings, oracle_families
from utils.experiment_logger import ExperimentLogger


_WORKER = {}


def select_source_rows(identity, parent_manifest, selection):
    """Select every original train frame except the reserved development parents."""
    parents = parent_manifest["parent_by_episode"]
    train_episodes = identity["splits"]["train"]
    reserved_episodes = selection["splits"]["validation"]
    if len(set(train_episodes)) != len(train_episodes) or len(reserved_episodes) != 40:
        raise ValueError("Expected unique original TRAIN episodes and 40 development validation episodes")
    train_parents = {parents[episode] for episode in train_episodes}
    reserved_parents = {parents[episode] for episode in reserved_episodes}
    if len(train_parents) != 630 or len(reserved_parents) != 40 or not reserved_parents <= train_parents:
        raise ValueError("Expected 630 TRAIN parents and 40 reserved development parents")
    selected_parents = train_parents - reserved_parents
    rows = []
    seen = set()
    for frame in identity["frames"]:
        sample_id = frame["id"]
        path = Path(sample_id)
        if path.is_absolute() or ".." in path.parts or sample_id in seen:
            raise ValueError(f"Invalid or duplicate source frame ID: {sample_id}")
        seen.add(sample_id)
        episode = str(path.parent)
        if episode in train_episodes and parents[episode] in selected_parents:
            rows.append({"sample_id": sample_id, "parent_id": parents[episode], "sha256": frame["sha256"]})
    if {row["parent_id"] for row in rows} != selected_parents:
        raise ValueError("Not every selected parent has source frames")
    return rows


def initialize_worker(config):
    """Allocate a detector once per worker; each worker uses bounded CPU threads."""
    import cv2
    from visual.corner_detection import Parameters, TorchCornerDetector

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _WORKER["detector"] = TorchCornerDetector(config["detector"]["device"],
                                               Parameters(**config["detector"].get("parameters", {})))
    _WORKER["root"] = Path(config["source_root"]).resolve()
    _WORKER["camera"] = config["camera"]
    _WORKER["certification"] = CertificationSettings(**config.get("certification", {}))


def inspect_frame(row):
    """Keep original detections and return explicit GT-matched corner IDs."""
    path = _WORKER["root"] / row["sample_id"]
    if not path.is_file() or file_sha256(path) != row["sha256"]:
        raise ValueError(f"Raw frame missing or hash mismatch: {path}")
    observation, gt = read_frame(path, _WORKER["camera"])
    result = _WORKER["detector"](observation["rgb"], observation["depth"],
                                 observation["intrinsic_cv"], observation["extrinsic_cv"])
    xyz, uv, object_ids = (value.cpu().numpy() for value in (result.xyz, result.uv, result.object_id))
    limits = _WORKER["certification"]
    peg_ids = peg_corner_ids(xyz, uv, object_ids, gt, observation,
                             limits.coordinate_tolerance_m, limits.projection_tolerance_px)
    family, rim_ids, _ = oracle_families(xyz, uv, object_ids, observation, gt, _WORKER["certification"])
    return {**row, **physical_vertex_conditions(peg_ids, family, rim_ids)}


def initialize_cached_worker(config):
    """Keep raw-frame paths and certification settings local to each worker."""
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _WORKER["root"] = Path(config["source_root"]).resolve()
    _WORKER["cache_root"] = Path(config["cache_root"]).resolve()
    _WORKER["camera"] = config["camera"]
    _WORKER["certification"] = CertificationSettings(**config.get("certification", {}))


def inspect_cached_frame(row):
    """Use checksum-verified detector XYZ and reconstruct its camera pixels."""
    import numpy as np
    from data_collection.vertex_family_labels import project_world

    path = _WORKER["root"] / row["sample_id"]
    cache_path = _WORKER["cache_root"] / row["cache_file"]
    if not path.is_file() or file_sha256(path) != row["sha256"]:
        raise ValueError(f"Raw frame missing or hash mismatch: {path}")
    if not cache_path.is_file() or file_sha256(cache_path) != row["cache_sha256"]:
        raise ValueError(f"Cached graph missing or hash mismatch: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as graph:
        valid = graph["node_valid"][0]
        xyz = graph["xyz_world"][0, valid]
        object_ids = graph["object_type"][0, valid]
    observation, gt = read_frame(path, _WORKER["camera"])
    uv, _ = project_world(xyz, observation["intrinsic_cv"], observation["extrinsic_cv"])
    limits = _WORKER["certification"]
    peg_ids = peg_corner_ids(xyz, uv, object_ids, gt, observation,
                             limits.coordinate_tolerance_m, limits.projection_tolerance_px)
    family, rim_ids, _ = oracle_families(xyz, uv, object_ids, observation, gt, limits)
    source = {key: row[key] for key in ("sample_id", "parent_id", "sha256")}
    return {**source, **physical_vertex_conditions(peg_ids, family, rim_ids)}


def iter_frame_decisions(rows, config, initializer, inspector):
    """Yield decisions in source order from isolated, bounded-thread workers."""
    with ProcessPoolExecutor(
        max_workers=config.get("workers", 4),
        initializer=initializer,
        initargs=(config,),
        mp_context=torch.multiprocessing.get_context("spawn"),
    ) as pool:
        yield from pool.map(inspector, rows, chunksize=8)


def prepare(config):
    """Select original detector pixels with the existing raw-frame entrypoint."""
    return prepare_selection(config, use_cache=False)


def prepare_selection(config, *, use_cache):
    """Record per-frame decisions and counts; this never alters the source dataset."""
    workers = config.get("workers", 4)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    paths = {name: Path(config[name]) for name in ("source_identity", "source_parent_manifest", "development_selection")}
    identity, parents, selection = (json.loads(paths[name].read_text()) for name in paths)
    rows = select_source_rows(identity, parents, selection)
    if use_cache:
        cache_root = Path(config["cache_root"]).resolve()
        inventory_path = cache_root / "metadata/cache_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        entries = {entry["sample_id"]: entry for entry in inventory["files"]}
        if len(entries) != len(inventory["files"]):
            raise ValueError("Duplicate sample IDs in cache inventory")
        for row in rows:
            entry = entries[row["sample_id"]]
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Cache file must be relative to cache_root")
            row.update(cache_file=str(relative), cache_sha256=entry["sha256"])
    max_frames = config.get("max_frames")
    if max_frames is not None:
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be null or a positive integer")
        rows = rows[:max_frames]
    with ExperimentLogger(team="graph", config=config, seed=0,
                          model="physical_vertex_selection", job_type="data_preparation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "effective_config.json", config)
        provenance = {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()}
        if use_cache:
            provenance["cache_inventory"] = {"path": str(inventory_path), "sha256": file_sha256(inventory_path)}
            provenance["pixel_source"] = "projection of cached detector XYZ"
        provenance["certification"] = asdict(CertificationSettings(**config.get("certification", {})))
        provenance["code_sha256"] = {str(path): file_sha256(path) for path in (
            Path(__file__), Path("data_collection/physical_vertex_filter.py"),
            Path("data_collection/vertex_family_labels.py"))}
        if use_cache:
            alias_path = Path("data_collection/physical_vertex_cache_selection.py")
            provenance["code_sha256"][str(alias_path)] = file_sha256(alias_path)
        write_json(output / "provenance.json", provenance)
        totals = {"frames": 0, "peg_pass": 0, "hole_pass": 0, "both_pass": 0}
        started = time.monotonic()
        with (output / "frames.jsonl").open("w", buffering=1) as stream:
            initializer = initialize_cached_worker if use_cache else initialize_worker
            inspector = inspect_cached_frame if use_cache else inspect_frame
            for result in iter_frame_decisions(rows, config, initializer, inspector):
                stream.write(json.dumps(result, sort_keys=True) + "\n")
                totals["frames"] += 1
                for name in ("peg_pass", "hole_pass", "both_pass"):
                    totals[name] += result[name]
                if totals["frames"] % 200 == 0 or totals["frames"] == len(rows):
                    print(f"physical vertex selection {totals}, elapsed={time.monotonic()-started:.1f}s", flush=True)
        totals.update(parents=len({row["parent_id"] for row in rows}), selected_source_parents=590,
                      elapsed_seconds=time.monotonic() - started)
        # Only an unrestricted run can claim complete 590-parent coverage.
        totals["complete"] = max_frames is None and totals["frames"] == len(rows)
        write_json(output / "summary.json", totals)
        for name in ("effective_config.json", "provenance.json", "frames.jsonl", "summary.json"):
            logger.link_output(output / name)
        logger.record(**totals)
        print(f"Physical vertex selection: {output}", flush=True)
    return output
