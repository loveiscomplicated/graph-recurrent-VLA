"""Collect new episodes and replay a fixed corner detector comparison."""

import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from data_collection.episode_dataset import file_sha256
from data_collection.mixed_trajectory_collection import collect_episode
from data_collection.storage import write_json
from evaluation.box_exterior_recovery import initialize, score_episode
from evaluation.hole_rim_holdout import actor_ids, make_environment
from evaluation.pilot_corner_detection import sample_frames
from utils.experiment_logger import ExperimentLogger


def fresh_records(config, protocol):
    """Use fresh scene and action seeds with the locked protocol's mix settings."""
    templates = [row for row in protocol["records"] if row["split"] == "test"]
    records = [
        {
            **template,
            "seed": config["seed_start"] + i,
            "action_seed": config["action_seed_start"] + i,
            "split": "fresh",
        }
        for i, template in enumerate(templates[: config["seed_count"]])
    ]
    if len(records) != config["seed_count"]:
        raise ValueError("Not enough test protocol templates for fresh trajectories")
    prior_seeds = {row["seed"] for row in protocol["records"]}
    for manifest in config.get("exclude_manifests", []):
        inventory = json.loads(Path(manifest).read_text())
        rows = inventory.get("records", inventory.get("rows", []))
        if not rows:
            raise ValueError(f"No seed records found in {manifest}")
        prior_seeds.update(row["seed"] for row in rows)
    if prior_seeds & {row["seed"] for row in records}:
        raise ValueError("Fresh scene seeds overlap a prior evaluation inventory")
    return records


def collect_and_score(job):
    """Retain sampled observations and score both variants on those same files."""
    record, protocol, output, variant = job
    env = make_environment(record, protocol)
    inputs, scores, episodes = [], [], []
    try:
        with TemporaryDirectory(dir=output) as scratch:
            for mode in protocol["modes"]:
                receipt = collect_episode(env, record, mode, protocol, Path(scratch))
                source = Path(scratch) / receipt["path"]
                target = output / receipt["path"]
                target.mkdir()
                frames = []
                for frame in sample_frames(receipt):
                    if frame == 0:
                        continue
                    name = f"frame_{frame:06d}.npz"
                    shutil.copyfile(source / name, target / name)
                    frames.append(
                        {
                            "seed": record["seed"],
                            "mode": mode,
                            "frame": frame,
                            "path": str(Path(receipt["path"]) / name),
                            "sha256": receipt["frame_sha256"][frame],
                            "actor_ids": actor_ids(env),
                        }
                    )
                inputs.extend(frames)
                episodes.append({
                    key: receipt[key]
                    for key in ("seed", "mode", "steps", "initial_fingerprint")
                })
                shutil.rmtree(source)
                scores.extend(
                    score_episode((output, "trajectory", frames, variant))
                )
    finally:
        env.close()

    if len({episode["initial_fingerprint"] for episode in episodes}) != 1:
        raise ValueError(f"Initial state changed across modes for seed {record['seed']}")
    return inputs, scores, episodes



def evaluate(config):
    """Freeze the cohort and persist paired metrics by simulator grasp state."""
    protocol_path = Path(config["protocol"])
    protocol = json.loads(protocol_path.read_text())
    records = fresh_records(config, protocol)
    variant = config.get("comparison", "peg_plane_center")
    if variant not in ("peg_plane_center", "hole_depth_fallback", "hole_rim_replacement"):
        raise ValueError(f"Unsupported fresh comparison: {variant}")
    with ExperimentLogger(
        team="graph",
        model=f"{variant}_fresh",
        config=config,
        seed=config["seed_start"],
        job_type="evaluation",
    ) as logger, ProcessPoolExecutor(
        max_workers=config["workers"], initializer=partial(initialize, variant=variant)
    ) as pool:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        selection = output / "selection.json"
        write_json(selection, {
            "records": records,
            "protocol_sha256": file_sha256(protocol_path),
        })
        logger.link_output(selection)
        rows, inputs, episodes = [], [], []
        jobs = ((record, protocol, output, variant) for record in records)
        for frame_inputs, scores, receipts in pool.map(collect_and_score, jobs):
            inputs.extend(frame_inputs)
            rows.extend(scores)
            episodes.extend(receipts)
            print(f"{len(episodes)} trajectories, {len(rows)} evaluated frames", flush=True)

        source = output / "frames.json"
        write_json(source, {"rows": inputs, "episodes": episodes})
        logger.link_output(source)
        variants = ("baseline", variant)
        totals = {group: {name: Counter() for name in variants}
                  for group in ("all", "grasp", "non_grasp")}
        for row in rows:
            group = "grasp" if row["grasp"] else "non_grasp"
            for name in variants:
                totals[group][name].update(row[name])
                totals["all"][name].update(row[name])
        path = output / "trajectory.json"
        write_json(path, {
            "rows": rows,
            "totals": totals,
            "frames_sha256": file_sha256(source),
        })
        logger.link_output(path)
        logger.record(totals=totals, frames=len(rows), trajectories=len(episodes))
        print(path, json.dumps(totals), flush=True)
