"""Replay a development sample with the supported-geometry recovery option."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data_collection.storage import write_json
from evaluation.hole_rim_holdout import score_seed
from evaluation.hole_rim_recovery import initialize, pilot_frame, summarize
from utils.experiment_logger import ExperimentLogger


def initialize_geometry():
    """Enable the experimental variant within each evaluator worker."""
    initialize(compare_geometry=True)


def evaluate(config):
    """Compare all detector variants on unchanged development observations."""
    initialize_geometry()
    source = Path(config["source"])
    selection = json.loads((source / "selection.json").read_text())
    protocol = json.loads(Path(config["protocol"]).read_text())
    rows = json.loads((source / "trajectory.json").read_text())["rows"]
    with ExperimentLogger(
        team="graph",
        model="small_geometry",
        config=config,
        seed=protocol["generator"],
        job_type="evaluation",
    ) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        with ProcessPoolExecutor(
            max_workers=config["workers"], initializer=initialize_geometry
        ) as pool:
            resets = list(
                pool.map(
                    score_seed,
                    (
                        (record, protocol, "reset", output)
                        for record in selection["cohorts"]["reset"]
                    ),
                )
            )
            reset_rows = [row for reset in resets for row in reset]
            jobs = (
                (
                    source / row["path"],
                    row["sha256"],
                    {0: 18, 1: 19},
                    {
                        key: row[key]
                        for key in ("seed", "mode", "frame", "path", "sha256")
                    },
                )
                for row in rows
            )
            trajectory_rows = list(pool.map(pilot_frame, jobs, chunksize=4))
        for cohort, evaluated in (
            ("reset", reset_rows),
            ("trajectory", trajectory_rows),
        ):
            totals = summarize(evaluated)
            path = output / f"{cohort}.json"
            write_json(path, {"totals": totals, "rows": evaluated})
            logger.link_output(path)
            logger.record(**{cohort: totals})
            print(path, json.dumps(totals), flush=True)
