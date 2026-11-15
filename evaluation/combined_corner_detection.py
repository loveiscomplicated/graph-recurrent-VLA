"""Remeasure the fixed Peg, Box exterior and Hole rim pipeline together."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.box_exterior_recovery import initialize, score_episode
from utils.experiment_logger import ExperimentLogger


VARIANT = "hole_rim_replacement"


def score_frames(job):
    """Reuse the verified rim replay, adding frame provenance to each score."""
    root, cohort, rows = job
    rows = sorted(rows, key=lambda row: row["frame"])
    results = score_episode((root, cohort, rows, VARIANT))
    return [
        {
            **source,
            "grasp": result["grasp"],
            "metrics": result[VARIANT],
        }
        for source, result in zip(rows, results, strict=True)
    ]


def summarize(rows):
    """Aggregate corner counts and retain frame and independent episode counts."""
    counts = Counter()
    for row in rows:
        counts.update(row["metrics"])
    return {
        "frames": len(rows),
        "seeds": len({row["seed"] for row in rows}),
        "episodes": len({(row["seed"], row.get("mode")) for row in rows}),
        **counts,
    }


def evaluate(config):
    """Replay all three saved cohorts with current fixed recovery settings."""
    inputs = {cohort: Path(path) for cohort, path in config["sources"].items()}
    source_hashes = {str(path): file_sha256(path) for path in inputs.values()}
    settings = {
        **config,
        "rim_replay": VARIANT,
        "recover_peg_rear_end": True,
        "device": "cpu",
        "source_sha256": source_hashes,
    }
    with ExperimentLogger(
        team="visual", model="combined_corner_detection", config=settings,
        job_type="evaluation", seed=0,
    ) as logger, ProcessPoolExecutor(
        max_workers=config["workers"],
        initializer=partial(initialize, VARIANT, recover_peg_rear_end=True),
    ) as pool:
        scores, totals = {}, {}
        for cohort, manifest in inputs.items():
            data = json.loads(manifest.read_text())
            rows = data["cohorts"]["reset"] if cohort == "reset" else data["rows"]
            by_episode = defaultdict(list)
            for row in rows:
                by_episode[row["seed"], row.get("mode")].append(row)
            mode = "reset" if cohort == "reset" else "trajectory"
            jobs = [(manifest.parent, mode, group) for group in by_episode.values()]
            scores[cohort] = [row for batch in pool.map(score_frames, jobs) for row in batch]
            totals[cohort] = summarize(scores[cohort])
            print(cohort, totals[cohort], flush=True)

        trajectory = scores["trajectory"]
        totals["trajectory_grasp"] = summarize([row for row in trajectory if row["grasp"]])
        totals["trajectory_by_policy"] = {
            mode: summarize([row for row in trajectory if row["mode"] == mode])
            for mode in sorted({row["mode"] for row in trajectory})
        }
        inserts = [row for row in scores["insertion_replay"] if row["phase"] == "insert"]
        last = {}
        for row in inserts:
            last[row["seed"], row["mode"]] = row
        totals["insertion"] = summarize(inserts)
        totals["last_insert"] = summarize(last.values())
        output = Path(config["output_root"]) / logger.run_id / "scores.json"
        output.parent.mkdir(parents=True)
        write_json(output, {"source_sha256": source_hashes, "rows": scores, "totals": totals})
        logger.link_output(output)
        logger.record(totals=totals)
        print(output, json.dumps(totals), flush=True)
