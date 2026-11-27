"""Evaluate ACT or Diffusion Policy on the shared paired reset seeds."""

import json
from pathlib import Path

from data_collection.episode_dataset import file_sha256
from evaluation.surface_mamba_controller import _load_eval_seeds, _wilson_interval
from utils.experiment_logger import ExperimentLogger, write


def evaluate(config):
    """Evaluate one immutable checkpoint and persist each completed rollout."""
    from visual.baseline.evaluate import evaluate_seeds

    checkpoint = Path(config["checkpoint"])
    seeds_path = Path(config["seeds_file"])
    if file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint SHA-256 differs from config: {checkpoint}")
    if file_sha256(seeds_path) != config["seeds_sha256"]:
        raise ValueError(f"Seed file SHA-256 differs from config: {seeds_path}")
    seeds = _load_eval_seeds(seeds_path, config.get("seed_limit"))
    candidate = config["candidate"]
    output = Path(config["output"])
    effective_path = output / "effective_config.json"
    trace_path = output / "rollouts.jsonl"
    summary_path = output / "closed_loop_summary.json"
    if output.exists():
        if summary_path.exists() or json.loads(effective_path.read_text()) != config:
            raise ValueError(f"Resume requires an incomplete run with the same config: {output}")
    else:
        output.mkdir(parents=True)
        write(effective_path, config)

    rows = {}
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            row = json.loads(line)
            seed = row["seed"]
            if row["candidate"] != candidate or seed not in seeds or seed in rows:
                raise ValueError(f"Unexpected or duplicate rollout for seed {seed}")
            rows[seed] = row
    pending = [seed for seed in seeds if seed not in rows]
    with ExperimentLogger(
        team="visual",
        model=candidate,
        job_type="evaluate",
        config=config,
        seed=seeds[0],
        root=output / "experiments",
    ) as logger:
        logger.link_output(checkpoint, checkpoint=True)
        if pending:
            with trace_path.open("a") as trace:
                def save_row(result):
                    row = {"candidate": candidate, **result}
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    trace.flush()
                    rows[row["seed"]] = row

                task = (str(checkpoint), pending, set(), None, {}, False)
                evaluate_seeds(
                    task,
                    image_size=config["image_size"],
                    horizon=config["horizon"],
                    observation_device="cuda",
                    motion=config.get("motion"),
                    seed_inference=True,
                    on_result=save_row,
                )
        if set(rows) != set(seeds):
            raise RuntimeError("The rollout trace does not cover the configured seed set")
        successes = sum(row["success"] for row in rows.values())
        metrics = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "wilson_95": _wilson_interval(successes, len(rows)),
        }
        write(summary_path, {"run_id": logger.run_id, "summaries": {candidate: metrics}})
        logger.log({"success_rate": metrics["success_rate"], "successes": successes})
        logger.link_output(summary_path)
        print(f"{candidate}: {successes}/{len(rows)} successful rollouts", flush=True)
