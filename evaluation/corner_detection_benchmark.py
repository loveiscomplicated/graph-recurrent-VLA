"""Replay the frozen corner benchmark without collecting new observations."""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.hole_rim_recovery import initialize, pilot_frame, summarize
from utils.experiment_logger import ExperimentLogger


def replay_frame(job):
    """Apply the stored actor IDs and the cohort's original GT scoring rule."""
    root, cohort, row = job
    labels = {key: value for key, value in row.items() if key != "scores"}
    ids = {int(key): value for key, value in row["actor_ids"].items()}
    path = root / row["path"]
    return pilot_frame(
        (path, row["sha256"], ids, labels),
        front_only=cohort == "reset",
    )


def evaluate(config):
    """Score checksum-verified benchmark inputs using the selected detector options."""
    dataset = Path(config["dataset"])
    setup = partial(
        initialize,
        compare_geometry=True,
        compare_fragmented=config.get("compare_fragmented", False),
    )
    setup()
    with ExperimentLogger(
        team="graph",
        model="corner_detection_benchmark",
        config=config,
        seed=20261114,
        job_type="evaluation",
    ) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        with ProcessPoolExecutor(
            max_workers=config["workers"], initializer=setup
        ) as pool:
            for cohort in ("reset", "trajectory"):
                source = dataset / f"{cohort}.json"
                manifest = json.loads(source.read_text())
                jobs = ((dataset, cohort, row) for row in manifest["rows"])
                rows = list(pool.map(replay_frame, jobs, chunksize=4))
                totals = summarize(rows)
                result = output / source.name
                write_json(
                    result,
                    {
                        "cohort": cohort,
                        "frames": len(rows),
                        "totals": totals,
                        "rows": rows,
                        "source_sha256": file_sha256(source),
                    },
                )
                logger.link_output(result)
                logger.record(**{cohort: totals})
                print(result, json.dumps(totals), flush=True)
