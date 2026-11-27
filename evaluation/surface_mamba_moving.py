"""Evaluate Surface-Mamba under scripted motion and compare matched rollouts."""

import json
import math
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.surface_mamba_motion import MovingSurfaceEpisode, make_motion_plan
from evaluation.surface_mamba_controller import _wilson_interval, evaluate_closed_loop


def _verify_candidate_checkpoints(config):
    """Verify legacy single-candidate or named multi-candidate checkpoint hashes."""
    actual = {
        name: [file_sha256(Path(path)) for path in spec["checkpoints"]]
        for name, spec in config["candidates"].items()
    }
    expected = config["checkpoint_sha256"]
    if isinstance(expected, list):
        if set(actual) != {"ensemble"} or actual["ensemble"] != expected:
            raise ValueError("Moving evaluation checkpoint digest differs")
    elif actual != expected:
        raise ValueError("Moving evaluation candidate checkpoint digests differ")
    return actual


def evaluate(config):
    """Reuse the fixed controller evaluator with an exogenous box trajectory."""
    _verify_candidate_checkpoints(config)
    if file_sha256(Path(config["seeds_file"])) != config["seeds_sha256"]:
        raise ValueError("Moving evaluation seed file changed")
    evaluate_closed_loop(
        config,
        wrap_environment=lambda env: MovingSurfaceEpisode(
            env, config["motion"], config["horizon"],
        ),
    )


def read_rollouts(directories, seeds, motion, horizon):
    """Require exactly one verified rollout per reserved seed."""
    rows = {}
    expected = set(seeds)
    for directory in directories:
        directory = Path(directory)
        summary = json.loads((directory / "closed_loop_summary.json").read_text())
        trace_path = directory / "rollouts.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        metrics = summary["summaries"]["ensemble"]
        if (metrics["episodes"], metrics["successes"]) != (
            len(trace), sum(row["success"] for row in trace),
        ):
            raise ValueError(f"Rollout summary differs from trace: {trace_path}")
        for row in trace:
            seed = row["seed"]
            if seed not in expected or seed in rows or row["candidate"] != "ensemble":
                raise ValueError(f"Unexpected or duplicate evaluation seed: {seed}")
            plan = make_motion_plan(seed, motion, horizon)
            if row["trajectory_sha256"] != plan.sha256:
                raise ValueError(f"Incorrect motion plan evaluated at seed {seed}")
            expected_translation = plan.offsets[row["steps"]]
            if not np.allclose(
                row["realized_translation_m"],
                expected_translation,
                atol=2e-6,
                rtol=0,
            ):
                raise ValueError(f"Incorrect terminal box displacement at seed {seed}")
            rows[seed] = row
    if set(rows) != expected:
        raise ValueError(f"Missing {len(expected - rows.keys())} reserved evaluation seeds")
    return rows


def paired_comparison(baseline, adapted, motion, horizon):
    """Report paired gains and exact McNemar p-value across the same motion plans."""
    if set(baseline) != set(adapted) or not baseline:
        raise ValueError("Paired comparison requires identical nonempty seed sets")
    count = len(baseline)
    for seed in baseline:
        identity = ("trajectory_sha256", "control_frequency_hz")
        if any(baseline[seed][key] != adapted[seed][key] for key in identity):
            raise ValueError(f"Motion or control frequency differs at seed {seed}")
    before = sum(row["success"] for row in baseline.values())
    after = sum(row["success"] for row in adapted.values())
    gained = sum(
        adapted[seed]["success"] and not row["success"]
        for seed, row in baseline.items()
    )
    lost = sum(
        row["success"] and not adapted[seed]["success"]
        for seed, row in baseline.items()
    )
    discordant = gained + lost
    smaller_discordant_count = min(gained, lost)
    lower_tail_probability = sum(
        math.comb(discordant, count)
        for count in range(smaller_discordant_count + 1)
    ) / 2**discordant
    p_value = min(1.0, 2 * lower_tail_probability)
    return {
        "episodes_per_candidate": count,
        "baseline": {
            "successes": before,
            "rate": before / count,
            "wilson_95": _wilson_interval(before, count),
        },
        "adapted": {
            "successes": after,
            "rate": after / count,
            "wilson_95": _wilson_interval(after, count),
        },
        "delta_percentage_points": 100 * (after - before) / count,
        "adapted_only_success": gained,
        "baseline_only_success": lost,
        "mcnemar_exact_p": p_value,
        "motion_completed": {
            "baseline": sum(row["motion_completed"] for row in baseline.values()),
            "adapted": sum(row["motion_completed"] for row in adapted.values()),
        },
        "motion": motion,
        "horizon": horizon,
    }
