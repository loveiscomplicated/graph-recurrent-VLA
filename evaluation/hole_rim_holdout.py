"""Evaluate fixed Hole recovery variants on unused mixed-trajectory seeds."""

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from data_collection.episode_dataset import file_sha256
from data_collection.mixed_trajectory_collection import (
    camera_sensors,
    collect_episode,
    validate_camera,
)
from data_collection.storage import save_frame, write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.hole_rim_recovery import (
    evaluate_pair,
    initialize,
    pilot_frame,
    summarize,
)
from evaluation.pilot_corner_detection import sample_frames
from utils.experiment_logger import ExperimentLogger


def make_environment(record, protocol):
    return make_peg_insertion_env(
        seed=record["seed"],
        device="cpu",
        obs_mode="rgb+depth+segmentation",
        sensor_configs=camera_sensors(protocol["camera"]),
        control_mode=protocol["control_mode"],
        max_episode_steps=protocol["max_steps"],
    )


def reset_frame(env, seed, protocol, output=None):
    """Render an independent reset and score both variants on the same input."""
    observation, _ = env.reset(seed=seed)
    validate_camera(observation, protocol["camera"])
    sensor = {
        name: value[0].cpu().numpy()
        for name, value in observation["sensor_data"]["base_camera"].items()
    }
    sensor.update(
        {
            name: value[0].cpu().numpy()
            for name, value in observation["sensor_param"]["base_camera"].items()
        }
    )
    gt = {
        name: value[0].cpu().numpy() for name, value in vars(get_gt_state(env)).items()
    }
    labels = {"seed": seed, "frame": 0}
    if output is not None:
        path = output / f"reset_{seed}.npz"
        save_frame(path, observation, ground_truth=vars(get_gt_state(env)))
        labels.update(
            path=path.name, sha256=file_sha256(path), actor_ids=actor_ids(env)
        )
    return {
        **labels,
        "scores": evaluate_pair(sensor, gt, actor_ids(env), front_only=True),
    }


def actor_ids(env):
    task = env.unwrapped
    return {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}


def score_seed(job):
    """Run one reset or one seed's three unfiltered rollouts."""
    record, protocol, cohort, output, *options = job
    save_reset = options[0] if options else False
    seed = record["seed"]
    env = make_environment(record, protocol)
    try:
        if cohort == "reset":
            return [reset_frame(env, seed, protocol, output if save_reset else None)]
        rows = []
        fingerprints = set()
        with TemporaryDirectory(dir=output) as scratch:
            for mode in protocol["modes"]:
                receipt = collect_episode(env, record, mode, protocol, Path(scratch))
                fingerprints.add(receipt["initial_fingerprint"])
                source = Path(scratch) / receipt["path"]
                destination = output / receipt["path"]
                destination.mkdir()
                for frame in sample_frames(receipt):
                    if frame == 0:
                        continue
                    name = f"frame_{frame:06d}.npz"
                    path = destination / name
                    shutil.copyfile(source / name, path)
                    labels = {
                        "seed": seed,
                        "mode": mode,
                        "frame": frame,
                        "steps": receipt["steps"],
                        "path": str(path.relative_to(output)),
                        "sha256": receipt["frame_sha256"][frame],
                        "actor_ids": actor_ids(env),
                    }
                    rows.append(
                        pilot_frame((path, labels["sha256"], actor_ids(env), labels))
                    )
                shutil.rmtree(source)
        if len(fingerprints) != 1:
            raise ValueError(
                f"Initial observations differ across modes for seed {seed}"
            )
        return rows
    finally:
        env.close()


def select_records(config, protocol):
    """Freeze selection from the protocol before running either detector."""
    excluded = set()
    for path in config["tuning_results"]:
        rows = json.loads(Path(path).read_text())["rows"]
        excluded.update(row["seed"] for row in rows)
    excluded.update(r["seed"] for r in protocol["records"] if r["split"] == "pilot")
    cohorts = {}
    for cohort, settings in config["cohorts"].items():
        records = [r for r in protocol["records"] if r["split"] == settings["split"]]
        count = settings["count"]
        chosen = [record for record in records if record["seed"] not in excluded][
            :count
        ]
        seeds = {r["seed"] for r in chosen}
        if len(seeds) != count or seeds & excluded:
            raise ValueError(
                f"{cohort} must have {count} distinct seeds disjoint from tuning and other cohorts"
            )
        excluded.update(seeds)
        cohorts[cohort] = chosen
    return cohorts


def evaluate(config):
    """Record fixed holdout inputs, paired frame results, and aggregate counts."""
    protocol_path = config["protocol"]
    protocol = json.loads(Path(protocol_path).read_text())
    cohorts = select_records(config, protocol)
    compare_geometry = config.get("compare_geometry", False)
    save_reset = config.get("save_reset", False)
    initialize(compare_geometry=compare_geometry)
    with ExperimentLogger(
        team="graph",
        model="hole_rim_holdout",
        config=config,
        seed=protocol["generator"],
        job_type="evaluation",
    ) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        sources = [protocol_path, *config["tuning_results"]]
        selection = {
            "cohorts": cohorts,
            "source_sha256": {path: file_sha256(path) for path in sources},
        }
        write_json(output / "selection.json", selection)
        logger.link_output(output / "selection.json")
        with ProcessPoolExecutor(
            max_workers=config["workers"],
            initializer=partial(initialize, compare_geometry=compare_geometry),
        ) as pool:
            for cohort, records in cohorts.items():
                rows = []
                jobs = [
                    (record, protocol, cohort, output, save_reset) for record in records
                ]
                for index, result in enumerate(pool.map(score_seed, jobs), start=1):
                    rows.extend(result)
                    if index % 10 == 0 or index == len(jobs):
                        print(
                            f"{cohort}: {index}/{len(jobs)} seeds, {len(rows)} frames",
                            flush=True,
                        )
                totals = summarize(rows)
                path = output / f"{cohort}.json"
                write_json(
                    path,
                    {
                        "cohort": cohort,
                        "frames": len(rows),
                        "totals": totals,
                        "rows": rows,
                    },
                )
                logger.link_output(path)
                logger.record(**{cohort: totals})
                print(path, json.dumps(totals), flush=True)
