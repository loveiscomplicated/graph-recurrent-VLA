"""Prepare certified corner episodes concurrently without modifying their source.

The established serial adapter does not expose a frame/episode preparation API.
This separate workflow preserves its byte checks and output contract while the
shared extraction interface remains unchanged.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.temporal.corner_tracks import Episode, FORMAT, Frame, write_episode
from data_collection.vertex_family_manifest import read_manifest
from graph.temporal.corner_tracking.tracker import Observation
from utils.experiment_logger import ExperimentLogger, clean, write


def plan_episodes(manifest, seed, fraction):
    """Use the serial adapter's exact parent reservation and chronological checks."""
    selection = manifest.get("selection", {})
    if "frames_per_parent" not in selection or selection["frames_per_parent"] is not None:
        raise ValueError("Source must explicitly declare full episodes with selection.frames_per_parent=null")
    if type(fraction) not in (float, int) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("calibration_fraction must be finite and strictly between zero and one")
    parents = sorted({row["parent_id"] for row in manifest["samples"] if row["split"] == "train"})
    np.random.default_rng(seed).shuffle(parents)
    count = max(1, int(len(parents) * fraction))
    if count >= len(parents):
        raise ValueError("Need at least two training parents to reserve calibration")
    calibration_parents = set(parents[:count])
    grouped = {}
    for row in manifest["samples"]:
        grouped.setdefault(str(Path(row["sample_id"]).parent), []).append(row)
    result = []
    for name, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["sample_id"])
        expected = [f"{name}/frame_{index:06d}.npz" for index in range(len(rows))]
        if [row["sample_id"] for row in rows] != expected:
            raise ValueError(f"Need contiguous full episodes, not selected frames: {name}")
        split = rows[0]["split"]
        parent = rows[0]["parent_id"]
        if any(row["split"] != split or row["parent_id"] != parent for row in rows):
            raise ValueError("An episode cannot change split or parent")
        if split == "train" and parent in calibration_parents:
            split = "calibration"
        result.append(dict(episode=name, parent=parent, split=split, rows=rows))
    return result


def source_paths(root, row):
    """Check source containment and all three source checksums before reading."""
    paths = {}
    for name, checksum in (("graph_path", "graph_sha256"), ("labels_path", "labels_sha256"),
                           ("raw_path", "source_sha256")):
        path = (root / row[name]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != row[checksum]:
            raise ValueError(f"Family source path/hash mismatch: {row['sample_id']}")
        paths[name] = path
    return paths


def prepare_frame(root, row, camera):
    """Apply the serial adapter's graph, robot, GT and certification checks."""
    from data_collection.episode_dataset import read_frame
    from data_collection.physical_vertex_filter import peg_corner_ids
    from graph.utils.object21_targets import object21_from_gt
    from training.gnn.data import load_cached_sample
    from visual.corner_detection.evaluation import corners

    paths = source_paths(root, row)
    sample = load_cached_sample(paths["graph_path"], row["sample_identity"], row["sample_id"])
    graph = sample.graph
    keep = graph.node_valid[0].numpy()
    xyz = graph.xyz_world[0, keep].numpy()
    kind = graph.object_type[0, keep].numpy()
    observation, ground_truth = read_frame(paths["raw_path"], camera)
    np.testing.assert_allclose(sample.targets["object21"].numpy(), object21_from_gt(ground_truth).numpy(),
                               atol=1e-6, rtol=1e-5,
                               err_msg="Cached Object21 target differs from the paired raw GT")
    with np.load(paths["raw_path"], allow_pickle=False) as raw:
        robot = np.concatenate((raw["observation/extra/tcp_pose"][0],
                                raw["observation/agent/qpos"][0, -2:])).astype(np.float32)
    if not np.allclose(robot[:3], graph.tcp_xyz_world[0].numpy(), atol=1e-6, rtol=0):
        raise ValueError("Observed robot position differs from cached TCP")
    with np.load(paths["labels_path"], allow_pickle=False) as labels:
        np.testing.assert_array_equal(labels["xyz_world"], xyz)
        identities = np.full(len(xyz), -1, dtype=np.int64)
        peg = peg_corner_ids(xyz, labels["uv"], kind, ground_truth, observation)
        identities[kind == 0] = peg[kind == 0]
        box_known = (kind == 1) & (labels["physical_id"] >= 0)
        identities[box_known] = labels["physical_id"][box_known] + 8
    references = corners(ground_truth)
    observed = Observation(xyz, kind, graph.confidence[0, keep].numpy(),
                           graph.confidence_valid[0, keep].numpy(), robot)
    return Frame(row["sample_id"], observed, identities,
                 np.asarray([reference["xyz"] for reference in references], dtype=np.float32),
                 np.asarray([reference["object_id"] for reference in references], dtype=np.int64),
                 sample.targets["object21"][0].numpy())


def prepare_episode(task):
    """Write one isolated episode atomically and return its manifest row."""
    import torch

    torch.set_num_threads(1)
    index, source, destination, camera, plan = task
    root = Path(source)
    frames = [prepare_frame(root, row, camera) for row in plan["rows"]]
    episode = Episode(plan["episode"], plan["parent"], plan["split"], frames)
    filename = f"episode_{index:06d}.npz"
    destination = Path(destination)
    path = destination / filename
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing episode: {path}")
    temporary = destination / f"episode_{index:06d}.partial.npz"
    write_episode(temporary, episode)
    temporary.replace(path)
    return dict(episode=episode.name, parent=episode.parent, split=episode.split,
                frames=len(frames), file=filename, sha256=file_sha256(path))


def prepare(config):
    """Prepare independent episodes in parallel and publish a manifest on success."""
    source = Path(config["source"]["family_dataset"]).expanduser().resolve()
    manifest = read_manifest(source)
    workers = config.get("workers", 4)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    seed = config.get("seed", 0)
    plans = plan_episodes(manifest, seed, config.get("calibration_fraction", 0.1))
    camera = config.get("camera", "base_camera")
    total_frames = sum(len(plan["rows"]) for plan in plans)
    with ExperimentLogger(team="graph", config=config, seed=seed,
                          model="corner_tracking_data", job_type="data_preparation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write(output / "preparation_plan.json", dict(source_dataset_sha256=manifest["dataset_sha256"],
                                                       episodes=len(plans), frames=total_frames,
                                                       workers=workers, status="incomplete_until_manifest"))
        logger.link_output(output / "preparation_plan.json")
        print(f"Preparing {len(plans)} episodes / {total_frames} frames with {workers} workers: {output}",
              flush=True)
        start = time.monotonic()
        completed_frames = 0
        results = {}
        tasks = [(index, str(source), str(output), camera, plan) for index, plan in enumerate(plans)]
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(prepare_episode, task): task[0] for task in tasks}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        row = future.result()
                    except Exception:
                        # A corrupt early frame must not leave hundreds of queued
                        # episodes running before the failure can be reported.
                        for pending in futures:
                            pending.cancel()
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
                    results[index] = row
                    completed_frames += row["frames"]
                    if len(results) % 10 == 0 or len(results) == len(plans):
                        print(f"Prepared {len(results)}/{len(plans)} episodes, "
                              f"{completed_frames}/{total_frames} frames in {time.monotonic() - start:.1f}s",
                              flush=True)
        except Exception:
            print(f"Preparation failed; partial files remain in {output}; no manifest was published.", flush=True)
            raise
        preparation_sha256 = file_sha256(Path(__file__))
        provenance = dict(source_dataset_sha256=manifest["dataset_sha256"], config=clean(config),
                          preparation_source_sha256=preparation_sha256,
                          certification="existing family BOX; PEG coordinate/projection")
        final = dict(format=FORMAT, time_unit="saved_frame", episodes=[results[index] for index in range(len(plans))],
                     provenance=provenance)
        final["dataset_sha256"] = canonical_hash(final)
        # The existing reader treats the final manifest as the completion marker.
        # Its atomic write prevents an interrupted run looking complete.
        write(output / "manifest.json", final)
        logger.record(dataset_sha256=final["dataset_sha256"], frames=total_frames,
                      preparation_workers=workers, preparation_source_sha256=preparation_sha256)
        logger.link_output(output / "manifest.json")
        print(f"Corner tracking dataset: {output}", flush=True)
        return str(output)
