"""Run baseline rollouts, moving-data collection, fine-tuning, and paired replay."""

from collections import deque
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.surface_mamba_controller import _load_eval_seeds
from evaluation.surface_mamba_moving import paired_comparison, read_rollouts
from main import load_config
from utils.experiment_logger import ExperimentLogger


EAGER_CUDA_VALIDATION_ENV = "SURFACE_CONTEXT_EAGER_CUDA_VALIDATION"


def _record_config(path, config):
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"Configuration changed under an existing experiment: {path}")
    else:
        write_json(path, config)


def _child_environment(gpu, overrides=None):
    """Build a child environment without leaking collection recovery settings."""
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
    )
    environment.pop(EAGER_CUDA_VALIDATION_ENV, None)
    if overrides:
        environment.update(overrides)
    return environment


def _run_jobs(root, stage, jobs, gpu_ids, environment_overrides=None):
    """Run one child per GPU and stop siblings if a child fails."""
    stage_root = root / "jobs" / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    pending = deque()
    for name, config, receipt in jobs:
        config_path = stage_root / f"{name}.json"
        _record_config(config_path, config)
        if not receipt.is_file():
            pending.append((name, config_path, receipt))
    available = deque(gpu_ids)
    running = {}
    try:
        while pending or running:
            while pending and available:
                name, config_path, receipt = pending.popleft()
                gpu = available.popleft()
                log_path = stage_root / f"{name}.log"
                environment = _child_environment(gpu, environment_overrides)
                with log_path.open("a") as log:
                    process = subprocess.Popen(
                        [sys.executable, "main.py", "--config", str(config_path)],
                        stdout=log, stderr=subprocess.STDOUT, env=environment,
                    )
                running[process] = (gpu, log_path, receipt)
                print(f"{stage}: started {name} on GPU {gpu}", flush=True)
            for process, (gpu, log_path, receipt) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                del running[process]
                if code or not receipt.is_file():
                    raise RuntimeError(f"{stage} failed: {log_path}")
                available.append(gpu)
                print(f"{stage}: completed {log_path.stem}", flush=True)
            if running:
                sleep(1)
    finally:
        for process in running:
            process.terminate()
        for process in running:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _evaluation_jobs(root, phase, protocol, checkpoints, hashes, seeds):
    jobs = []
    position = protocol["seed_start"]
    group_size = (len(seeds) + 1) // 2
    for shard_index, offset in enumerate(range(0, len(seeds), group_size)):
        size = len(seeds[offset : offset + group_size])
        name = f"shard_{shard_index}"
        output = root / phase / name
        config = {
            **protocol,
            "trainer": "evaluation.surface_mamba_moving:evaluate",
            "output": str(output),
            "resume": True,
            "seed_start": position + offset,
            "seed_limit": size,
            "checkpoint_sha256": hashes,
            "comparison_reference": "ensemble",
            "candidates": {"ensemble": {"input": "graph", "checkpoints": checkpoints}},
        }
        jobs.append((name, config, output / "closed_loop_summary.json"))
    return jobs


def _collection_jobs(root, config, protocol, excluded):
    jobs = []
    occupied = set()
    for shard, buckets in enumerate(config["collection_shards"]):
        for bucket in buckets:
            start = bucket["seed_start"]
            seeds = set(range(start, start + bucket["max_attempts"]))
            if occupied & seeds:
                raise ValueError("Moving collection seed ranges overlap")
            occupied.update(seeds)
        output = root / "collection" / f"shard_{shard}"
        collection = {
            "trainer": "data_collection.surface_mamba_moving:collect",
            "output": str(output),
            "environment": {**protocol["environment"], "resolution": 512,
                            "control_mode": "pd_ee_delta_pose"},
            "frontend": protocol["frontend"],
            "horizon": protocol["horizon"],
            "motion": protocol["motion"],
            "buckets": buckets,
            "excluded_seeds": excluded,
        }
        jobs.append((f"shard_{shard}", collection, output / "collection_summary.json"))
    return jobs


def _prepare_collection_protocol(config):
    """Load the fixed frontend protocol and build the collection exclusion set."""
    fixed = load_config(config["fixed_evaluation_config"])
    protocol = {
        key: fixed[key]
        for key in (
            "frontend",
            "environment",
            "horizon",
            "seeds_file",
            "controller_device",
        )
    }
    protocol.update(
        motion=config["motion"],
        seeds_sha256=file_sha256(protocol["seeds_file"]),
    )

    tap_checkpoint = protocol["frontend"]["tapnextpp"]["checkpoint"]
    if file_sha256(tap_checkpoint) != config["tap_checkpoint_sha256"]:
        raise ValueError(
            "TAPNext++ checkpoint differs from the fixed-hole evaluation setup"
        )

    all_evaluation_seeds = _load_eval_seeds(protocol["seeds_file"])
    excluded = set(all_evaluation_seeds)
    fixed_dataset = Path(config["fixed_dataset"])
    for split in ("train", "validation", "test"):
        split_path = fixed_dataset / f"{split}.npz"
        with np.load(split_path, allow_pickle=False) as data:
            excluded.update(data["reset_seed"].tolist())

    return protocol, sorted(excluded)


def collect_dataset(config):
    """Collect and merge only the moving State36 dataset for a later experiment."""
    from data_collection.surface_mamba_moving import merge_collections

    root = Path(config["output"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = Path(config.get("dataset_output", root / "state36")).resolve()

    with (root / "collection_launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _record_config(root / "collection_effective_config.json", config)

        gpu_ids = config["gpu_ids"]
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError(
                "gpu_ids must be a nonempty list of distinct identifiers"
            )

        protocol, excluded = _prepare_collection_protocol(config)
        collection_jobs = _collection_jobs(root, config, protocol, excluded)
        _run_jobs(
            root,
            "collection",
            collection_jobs,
            gpu_ids,
            environment_overrides={EAGER_CUDA_VALIDATION_ENV: "1"},
        )

        manifest_path = dataset / "dataset_manifest.json"
        if not manifest_path.is_file():
            merge_collections(
                [receipt for _, _, receipt in collection_jobs],
                dataset,
                {**protocol, "excluded_seeds": excluded},
            )

            with ExperimentLogger(
                team="recurrent",
                model="surface_mamba_moving_state36",
                job_type="prepare",
                config=config,
                root=root / "dataset_experiments",
            ) as logger:
                logger.link_output(dataset / "train.npz", kind="trajectory")
                logger.link_output(
                    dataset / "validation.npz", kind="trajectory"
                )
                logger.link_output(manifest_path, kind="metrics")
                logger.record(
                    dataset_manifest_sha256=file_sha256(manifest_path),
                    train_episodes=config["dataset_counts"][0],
                    validation_episodes=config["dataset_counts"][1],
                )

        manifest = json.loads(manifest_path.read_text())
        actual_counts = [
            manifest["splits"]["train"]["episodes"],
            manifest["splits"]["validation"]["episodes"],
        ]
        if actual_counts != config["dataset_counts"]:
            raise ValueError(
                "Merged moving dataset counts differ: "
                f"expected {config['dataset_counts']}, got {actual_counts}"
            )
        print(json.dumps(manifest, indent=2), flush=True)


def _training_jobs(root, config, protocol, sources, hashes, dataset, seeds):
    base = load_config(config["base_training_config"])["base_training"]
    settings = deepcopy(base["training"])
    training = config["fine_tuning"]
    settings.update(training["settings"])
    settings["optimizer"]["learning_rate"] = training["learning_rate"]
    jobs = []
    for seed, (path, digest) in enumerate(zip(sources, hashes, strict=True)):
        output = root / "fine_tuning" / f"seed_{seed}"
        run = output / f"mamba_seed_{seed}"
        job = {
            **base,
            "trainer": "training.surface_mamba_moving:fine_tune",
            "training": settings,
            "dataset": str(dataset),
            "output": str(output),
            "seeds": [seed],
            "mixing": {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0]},
            "initial_checkpoint": path,
            "initial_checkpoint_sha256": digest,
            "dataset_contract": {
                "motion": protocol["motion"], "counts": config["dataset_counts"],
                "evaluation_seeds": seeds,
            },
        }
        jobs.append((f"seed_{seed}", job, run / "summary.json"))
    return jobs


def run(config):
    """Keep all experiment outputs under a new root and prevent duplicate launchers."""
    root = Path(config["output"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_experiment(config, root)


def _run_experiment(config, root):
    """Finish each phase before scheduling the next."""
    from data_collection.surface_mamba_moving import merge_collections

    _record_config(root / "effective_config.json", config)
    final = root / "paired_summary.json"
    if final.exists():
        print(final.read_text(), end="")
        return
    gpu_ids = config["gpu_ids"]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must be a nonempty list of distinct identifiers")
    fixed = load_config(config["fixed_evaluation_config"])
    sources = fixed["candidates"]["mixed_ensemble"]["checkpoints"]
    baseline_hashes = json.loads(Path(config["fixed_evaluation_metadata"]).read_text())[
        "checkpoint_sha256"
    ]["mixed_ensemble"]
    if len(sources) != 5 or [file_sha256(path) for path in sources] != baseline_hashes:
        raise ValueError("Fixed-hole ensemble differs from its completed evaluation")

    protocol = {key: fixed[key] for key in (
        "frontend", "environment", "horizon", "seeds_file", "controller_device",
    )}
    protocol.update(
        motion=config["motion"], seed_start=config["seed_start"],
        seeds_sha256=file_sha256(protocol["seeds_file"]),
    )
    tap_checkpoint = protocol["frontend"]["tapnextpp"]["checkpoint"]
    if file_sha256(tap_checkpoint) != config["tap_checkpoint_sha256"]:
        raise ValueError("TAPNext++ checkpoint differs from the fixed-hole evaluation setup")
    seeds = _load_eval_seeds(protocol["seeds_file"], config["seed_count"], config["seed_start"])
    all_evaluation_seeds = _load_eval_seeds(protocol["seeds_file"])
    excluded = set(all_evaluation_seeds)
    for split in ("train", "validation", "test"):
        with np.load(Path(config["fixed_dataset"]) / f"{split}.npz", allow_pickle=False) as data:
            excluded.update(data["reset_seed"].tolist())

    baseline_jobs = _evaluation_jobs(root, "baseline", protocol, sources, baseline_hashes, seeds)
    _run_jobs(root, "baseline", baseline_jobs, gpu_ids)
    baseline = read_rollouts(
        [receipt.parent for _, _, receipt in baseline_jobs], seeds, protocol["motion"], protocol["horizon"],
    )

    collection_jobs = _collection_jobs(root, config, protocol, sorted(excluded))
    # Long moving-hole episodes exercise recovery branches that exceed the
    # full-graph validation specialization limit. Eager validation uses the
    # same tensor implementation and is scoped to collection children only.
    _run_jobs(
        root,
        "collection",
        collection_jobs,
        gpu_ids,
        environment_overrides={EAGER_CUDA_VALIDATION_ENV: "1"},
    )
    dataset = root / "state36"
    if not (dataset / "dataset_manifest.json").is_file():
        merge_collections(
            [receipt for _, _, receipt in collection_jobs], dataset,
            {**protocol, "excluded_seeds": sorted(excluded)},
        )
    training_jobs = _training_jobs(root, config, protocol, sources, baseline_hashes, dataset, all_evaluation_seeds)
    _run_jobs(root, "fine_tuning", training_jobs, gpu_ids)
    adapted_paths = [str(receipt.with_name("best.pt")) for _, _, receipt in training_jobs]
    adapted_hashes = [json.loads(receipt.read_text())["best_sha256"] for _, _, receipt in training_jobs]
    adapted_jobs = _evaluation_jobs(root, "adapted", protocol, adapted_paths, adapted_hashes, seeds)
    _run_jobs(root, "adapted", adapted_jobs, gpu_ids)
    adapted = read_rollouts(
        [receipt.parent for _, _, receipt in adapted_jobs], seeds, protocol["motion"], protocol["horizon"],
    )

    result = paired_comparison(baseline, adapted, protocol["motion"], protocol["horizon"])
    result.update(
        seed_list_sha256=protocol["seeds_sha256"], baseline_checkpoint_sha256=baseline_hashes,
        adapted_checkpoint_sha256=adapted_hashes,
        dataset_manifest_sha256=file_sha256(dataset / "dataset_manifest.json"),
    )
    with ExperimentLogger(
        team="recurrent", model="surface_mamba_moving", job_type="compare",
        config=config, root=root / "experiments",
    ) as logger:
        write_json(final, result)
        logger.link_output(final)
    print(json.dumps(result, indent=2), flush=True)
