"""Apply the physical-coverage audit to a previously annotated validation set."""

import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256, read_frame
from data_collection.physical_vertex_filter import peg_corner_ids, physical_vertex_conditions
from data_collection.storage import write_json
from data_collection.vertex_family_manifest import read_manifest
from utils.experiment_logger import ExperimentLogger


def prepare(config):
    """Audit validation frames using original detector UV and stored Rim certificates."""
    root = Path(config["family_dataset"]).resolve()
    manifest = read_manifest(root)
    rows = [row for row in manifest["samples"] if row["split"] == "validation"]
    with ExperimentLogger(team="graph", config=config, seed=0,
                          model="physical_vertex_validation", job_type="data_preparation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        records = []
        for row in rows:
            raw_path = (root / row["raw_path"]).resolve()
            labels_path = (root / row["labels_path"]).resolve()
            if not raw_path.is_relative_to(root) or not labels_path.is_relative_to(root):
                raise ValueError("Validation paths must remain in the dataset root")
            if file_sha256(raw_path) != row["source_sha256"] or file_sha256(labels_path) != row["labels_sha256"]:
                raise ValueError(f"Validation raw or labels hash mismatch: {row['sample_id']}")
            observation, gt = read_frame(raw_path, config.get("camera", "base_camera"))
            with np.load(labels_path, allow_pickle=False) as archive:
                peg_ids = peg_corner_ids(archive["xyz_world"], archive["uv"], archive["object_type"], gt, observation)
                result = physical_vertex_conditions(peg_ids, archive["oracle_family"], archive["physical_id"])
            records.append(dict(sample_id=row["sample_id"], parent_id=row["parent_id"], **result))
        with (output / "frames.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        summary = dict(dataset_sha256=manifest["dataset_sha256"], frames=len(records),
                       parents=len({row["parent_id"] for row in records}),
                       **{name: sum(row[name] for row in records) for name in ("peg_pass", "hole_pass", "both_pass")})
        write_json(output / "summary.json", summary)
        write_json(output / "effective_config.json", config)
        logger.record(**summary)
        logger.link_output(output / "summary.json")
        logger.link_output(output / "frames.jsonl")
        print(f"Physical vertex validation: {output}; {summary}", flush=True)
    return output
