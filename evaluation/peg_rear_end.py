"""Replay rear-end recovery on checksum-verified insertion frames."""

import json
from collections import defaultdict
from pathlib import Path

import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.trajectory_rim_recovery import load_frame
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import score, visible_reference


def frame_metrics(detector, sensor, references):
    """Score one prediction set using the existing one-to-one 3 mm assignment."""
    result = detector(
        sensor["rgb"], sensor["depth"], sensor["intrinsic_cv"], sensor["extrinsic_cv"]
    )
    predictions = result.as_dict()["corners"]
    peg, box = score(predictions, references, [])
    return {
        "peg_matches": peg["matched_3mm"],
        "peg_visible": peg["visible_reference"],
        "peg_unmatched": peg["predictions"] - peg["matched_3mm"],
        "box_outer_matches": box["by_kind"].get("box_outer", {}).get("matched", 0),
        "hole_rim_matches": box["by_kind"].get("hole_rim", {}).get("matched", 0),
        "box_unmatched": box["predictions"] - box["matched_3mm"],
    }


def evaluate(config):
    """Compare the detector with and without rear-end recovery on the same frames."""
    torch.set_num_threads(config.get("threads", 1))
    source = Path(config["source"])
    manifest = json.loads(source.read_text())
    rows = manifest["rows"]
    last_insert = defaultdict(int)
    for row in rows:
        if row["phase"] == "insert":
            key = row["seed"], row["mode"]
            last_insert[key] = max(last_insert[key], row["frame"])

    detectors = {
        name: TorchCornerDetector(
            device="cpu", recover_peg_rear_end=enabled, **config["detector"]
        )
        for name, enabled in (("baseline", False), ("rear_end", True))
    }
    with ExperimentLogger(
        team="visual", model="peg_rear_end", config=config, job_type="evaluation",
        seed=manifest["config"]["seed_start"],
    ) as logger:
        scored = []
        for row in rows:
            sensor, gt, _ = load_frame(source.parent, row)
            actor_ids = {int(key): value for key, value in row["actor_ids"].items()}
            references = visible_reference(gt, sensor, actor_ids)
            scored.append({
                **{key: row[key] for key in ("seed", "mode", "frame", "phase", "sha256")},
                "last_insert": row["phase"] == "insert"
                and row["frame"] == last_insert[row["seed"], row["mode"]],
                **{
                    name: frame_metrics(detector, sensor, references)
                    for name, detector in detectors.items()
                },
            })
        cohorts = {
            "align": [row for row in scored if row["phase"] == "align"],
            "insert": [row for row in scored if row["phase"] == "insert"],
            "last_insert": [row for row in scored if row["last_insert"]],
        }
        totals = {}
        for group, frames in cohorts.items():
            totals[group] = {
                "frames": len(frames),
                **{
                    name: {
                        metric: sum(row[name][metric] for row in frames)
                        for metric in scored[0][name]
                    }
                    for name in detectors
                },
            }
        output = Path(config["output_root"]) / logger.run_id / "trajectory.json"
        output.parent.mkdir(parents=True)
        write_json(output, {
            "source": str(source), "source_sha256": file_sha256(source),
            "rows": scored, "totals": totals,
        })
        logger.link_output(output)
        logger.record(totals=totals)
        print(output, json.dumps(totals), flush=True)
