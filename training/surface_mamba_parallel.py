"""Run independent Surface-Mamba training seeds across visible GPUs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from training.surface_mamba_controller import _validate_config


@dataclass(frozen=True)
class TrainingJob:
    """One independently resumable controller-training process."""

    name: str
    config: dict[str, Any]

    @property
    def checkpoint(self) -> Path:
        seed = self.config["seeds"][0]
        return Path(self.config["output"]) / f"mamba_seed_{seed}" / "best.pt"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration mappings without modifying either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_jobs(config: dict[str, Any]) -> list[TrainingJob]:
    """Expand experiment seed lists into isolated one-seed training jobs."""
    base = config.get("base_training")
    experiments = config.get("experiments")
    if not isinstance(base, dict):
        raise ValueError("base_training must be a mapping")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a nonempty list")

    warm_start_sources = config.get("warm_start_sources", [])
    if not isinstance(warm_start_sources, list):
        raise ValueError("warm_start_sources must be a list when provided")
    sources_by_seed = {}
    for source in warm_start_sources:
        if not isinstance(source, dict):
            raise ValueError("Each warm-start source must be a mapping")
        seed = source.get("seed")
        path = source.get("checkpoint")
        digest = source.get("sha256")
        if (
            type(seed) is not int
            or seed < 0
            or not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("Warm-start sources need seed, checkpoint, and SHA-256")
        if seed in sources_by_seed:
            raise ValueError(f"Duplicate warm-start source seed: {seed}")
        sources_by_seed[seed] = source

    jobs = []
    seen_names = set()
    seen_outputs = set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment must be a mapping")
        name = experiment.get("name")
        seeds = experiment.get("seeds")
        experiment_output = experiment.get("output")
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError("Experiment names must contain letters, digits, or underscores")
        if not isinstance(experiment_output, str) or not experiment_output:
            raise ValueError(f"Experiment {name!r} needs a nonempty output path")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Experiment {name!r} needs a nonempty seed list")

        overrides = {
            key: value
            for key, value in experiment.items()
            if key not in {"name", "output", "seeds"}
        }
        for seed in seeds:
            job_name = f"{name}_seed_{seed}"
            job_output = str(Path(experiment_output) / f"seed_{seed}")
            if job_name in seen_names or job_output in seen_outputs:
                raise ValueError(f"Duplicate Surface-Mamba job identity: {job_name}")

            job_config = _deep_merge(base, overrides)
            job_config.update(
                {
                    "trainer": job_config.get(
                        "trainer", "training.surface_mamba_controller:train"
                    ),
                    "output": job_output,
                    "seeds": [seed],
                    # CUDA_VISIBLE_DEVICES remaps the assigned GPU to local cuda:0.
                    "device": "cuda:0",
                }
            )
            if sources_by_seed:
                if seed not in sources_by_seed:
                    raise ValueError(f"Warm-start source is missing for seed {seed}")
                source = sources_by_seed[seed]
                checkpoint = Path(source["checkpoint"])
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Warm-start checkpoint not found: {checkpoint}"
                    )
                if file_sha256(checkpoint) != source["sha256"]:
                    raise ValueError(
                        f"Warm-start checkpoint digest changed: {checkpoint}"
                    )
                job_config["initial_checkpoint"] = str(checkpoint)
                job_config["initial_checkpoint_sha256"] = source["sha256"]
            _validate_config(job_config)
            jobs.append(TrainingJob(job_name, job_config))
            seen_names.add(job_name)
            seen_outputs.add(job_output)
    return jobs


def _resolve_gpu_ids(configured: Any) -> list[str]:
    """Resolve physical GPU selectors while honoring parent visibility."""
    if configured == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpu_ids = [value.strip() for value in visible.split(",") if value.strip()]
        else:
            gpu_ids = [str(index) for index in range(torch.cuda.device_count())]
    elif isinstance(configured, list):
        gpu_ids = [str(value) for value in configured]
    else:
        raise ValueError("gpu_ids must be 'auto' or a list of GPU indices/UUIDs")

    if not gpu_ids or any(not gpu_id for gpu_id in gpu_ids):
        raise RuntimeError("No GPUs are available for Surface-Mamba training")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must contain distinct GPU selectors")
    return gpu_ids


def _prepare_launch_directory(
    config: dict[str, Any], jobs: list[TrainingJob]
) -> tuple[Path, dict[str, Path]]:
    """Persist the launcher and per-job effective configurations."""
    output = Path(config["output"]).resolve()
    effective_path = output / "effective_config.json"
    if output.exists():
        if not config.get("resume"):
            raise FileExistsError(f"Parallel training output already exists: {output}")
        if not effective_path.is_file() or json.loads(effective_path.read_text()) != config:
            raise ValueError("Parallel training resume requires an identical effective config")
    else:
        output.mkdir(parents=True)
        write_json(effective_path, config)

    config_paths = {}
    for job in jobs:
        job_directory = output / "jobs" / job.name
        job_directory.mkdir(parents=True, exist_ok=True)
        config_path = job_directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != job.config:
            raise ValueError(f"Training job config changed on resume: {job.name}")
        if not config_path.exists():
            write_json(config_path, job.config)
        config_paths[job.name] = config_path
    return output, config_paths


def _terminate_processes(running: dict[subprocess.Popen, tuple[str, TrainingJob, Any]]) -> None:
    """Terminate and reap every still-running child before returning control."""
    for process in running:
        if process.poll() is None:
            process.terminate()
    for process in running:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for _, _, log_stream in running.values():
        log_stream.close()


def launch(config: dict[str, Any]) -> None:
    """Schedule one independent training process per configured GPU."""
    jobs = _build_jobs(config)
    gpu_ids = _resolve_gpu_ids(config.get("gpu_ids", "auto"))
    output, config_paths = _prepare_launch_directory(config, jobs)
    repository_root = Path(__file__).resolve().parents[1]
    python = Path(config.get("python", sys.executable))
    if not python.is_absolute():
        python = repository_root / python
    if not python.is_file():
        raise FileNotFoundError(f"Configured Python executable does not exist: {python}")

    pending = list(jobs)
    available_gpus = list(gpu_ids)
    running: dict[subprocess.Popen, tuple[str, TrainingJob, Any]] = {}
    completed = []
    try:
        while pending or running:
            while pending and available_gpus:
                job = pending.pop(0)
                gpu_id = available_gpus.pop(0)
                job_directory = output / "jobs" / job.name
                log_stream = (job_directory / "train.log").open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
                environment["PYTHONUNBUFFERED"] = "1"
                command = [
                    str(python),
                    str(repository_root / "main.py"),
                    "--config",
                    str(config_paths[job.name]),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=repository_root,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                )
                running[process] = (gpu_id, job, log_stream)
                write_json(
                    job_directory / "status.json",
                    {"status": "running", "gpu_id": gpu_id, "pid": process.pid},
                )
                print(f"Started {job.name} on GPU {gpu_id}", flush=True)

            finished = [process for process in running if process.poll() is not None]
            if not finished:
                time.sleep(0.25)
                continue
            for process in finished:
                gpu_id, job, log_stream = running.pop(process)
                log_stream.close()
                status = {
                    "status": "complete" if process.returncode == 0 else "failed",
                    "gpu_id": gpu_id,
                    "returncode": process.returncode,
                    "checkpoint": str(job.checkpoint),
                }
                write_json(output / "jobs" / job.name / "status.json", status)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"Surface-Mamba job {job.name} failed; see "
                        f"{output / 'jobs' / job.name / 'train.log'}"
                    )
                if not job.checkpoint.is_file():
                    raise RuntimeError(
                        f"Surface-Mamba job {job.name} exited without {job.checkpoint}"
                    )
                completed.append(job.name)
                available_gpus.append(gpu_id)
                print(f"Completed {job.name} on GPU {gpu_id}", flush=True)
    except BaseException:
        for process, (gpu_id, job, _) in running.items():
            write_json(
                output / "jobs" / job.name / "status.json",
                {
                    "status": "terminated",
                    "gpu_id": gpu_id,
                    "pid": process.pid,
                    "reason": "another job failed or the launcher was interrupted",
                },
            )
        _terminate_processes(running)
        raise

    write_json(
        output / "summary.json",
        {
            "status": "complete",
            "gpu_ids": gpu_ids,
            "jobs": completed,
            "checkpoints": {job.name: str(job.checkpoint) for job in jobs},
        },
    )
