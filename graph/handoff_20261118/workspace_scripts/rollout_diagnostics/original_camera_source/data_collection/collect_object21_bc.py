"""Collect verified raw BC demonstrations with one on-disk copy per shard."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess

from data_collection.collect_dataset import load_config, shard_command, verify_shard
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger


def collect_shard(settings, output, index, count):
    """Reuse the raw collector and validate each completed shard."""
    name = f"shard_{index:06d}"
    directory = output / "shards" / name
    seed = settings["start_seed"] + index * settings["attempts_per_shard"]
    command = shard_command(settings, directory, count, seed)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if not directory.exists():
        with (output / "logs" / f"{name}.log").open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    verification = verify_shard(directory, settings, count)
    if any(not seed <= item < seed + settings["attempts_per_shard"] for item in verification["seeds"]):
        raise ValueError(f"Shard seed is outside its allocated range: {directory}")
    write_json(directory / "verification.json", verification)
    return directory


def collect(config):
    """Collect fresh 512x512 episodes and record their source identities."""
    settings = load_config(config["collection"])
    if settings["resolution"] != 512 or settings["visibility_mode"] != "ends_and_grasp":
        raise ValueError("Object21 BC requires native 512x512 RGB-D and ends_and_grasp")
    if settings["control_mode"] != "pd_ee_delta_pose" or settings.get("publish"):
        raise ValueError("Use pd_ee_delta_pose and publish=false for the raw collection")
    exclusion = json.loads(Path(config["excluded_seeds"]).read_text())
    graph_seeds = set(exclusion["graph_development_seeds"])
    original_seeds = set(exclusion["original_collection_seeds"])
    if len(graph_seeds) != 630 or len(original_seeds) != 900 or not graph_seeds <= original_seeds:
        raise ValueError("Expected the original 900 seeds and their 630 Graph development seeds")
    if settings["start_seed"] <= max(original_seeds):
        raise ValueError("Start seed must be above the original collection's seed range")
    workers = config.get("parallel_shards", settings["workers"])
    if type(workers) is not int or workers < 1:
        raise ValueError("parallel_shards must be a positive integer")

    output = Path(settings["output"]).resolve()
    effective_config = output / "effective_config.json"
    previous_run = None
    if output.exists():
        if (not config.get("resume") or (output / "dataset_manifest.json").exists()
                or json.loads(effective_config.read_text()) != settings):
            raise ValueError("Existing collection is complete or needs resume with identical settings")
        previous_run = json.loads((output / "run_identity.json").read_text())["run_id"]
    else:
        output.mkdir(parents=True)
        for name in ("shards", "logs"):
            (output / name).mkdir()
        write_json(effective_config, settings)
    counts = [
        min(settings["episodes_per_shard"], settings["num_episodes"] - offset)
        for offset in range(0, settings["num_episodes"], settings["episodes_per_shard"])
    ]
    with ExperimentLogger(team="integration", model="object21_bc_raw",
                          seed=settings["start_seed"], config=dict(config, effective=settings),
                          job_type="collection") as logger:
        if previous_run is not None:
            logger.record(source_run_id=previous_run)
        write_json(output / "run_identity.json", {"run_id": logger.run_id})
        episodes = []
        # Bound the number of active shards and check disk space between waves.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            start = 0
            while start < len(counts):
                stop = min(start + workers, len(counts))
                free_gib = shutil.disk_usage(output).free / 2**30
                estimate = config.get("disk_gib_per_episode", 0.066)
                reserve = config.get("disk_reserve_gib", 0.5)
                while stop > start:
                    new_count = sum(counts[index] for index in range(start, stop)
                                    if not (output / "shards" / f"shard_{index:06d}").exists())
                    if not new_count or free_gib >= reserve + estimate * new_count:
                        break
                    stop -= 1
                if stop == start:
                    raise OSError("Insufficient free disk for the next collection wave")
                jobs = [executor.submit(collect_shard, settings, output, index, counts[index])
                        for index in range(start, stop)]
                for job in jobs:
                    directory = job.result()
                    raw = json.loads((directory / "manifest.json").read_text())
                    for row in raw["episodes"]:
                        episodes.append(dict(row, path=f"shards/{directory.name}/{row['path']}"))
                    print(f"Verified {len(episodes)}/{settings['num_episodes']} episodes", flush=True)
                    write_json(output / "progress.json", {
                        "episodes": len(episodes), "target": settings["num_episodes"],
                    })
                start = stop
        seeds = {row["seed"] for row in episodes}
        if len(seeds) != len(episodes) or seeds & original_seeds:
            raise ValueError("New episodes contain duplicate or previously collected seeds")
        manifest = output / "dataset_manifest.json"
        write_json(manifest, {
            "run_id": logger.run_id, "episode_count": len(episodes), "episodes": episodes,
            "config": settings, "graph_dataset_sha256": exclusion["graph_dataset_sha256"],
            "original_collection_run_id": exclusion["original_collection_run_id"],
            "frames": sum(row["observations"] for row in episodes),
        })
        logger.link_output(manifest, kind="episode_metadata")
        logger.log({"episodes": len(episodes)})
