"""Build a graph-only M0 dataset from audited physical-coverage frame IDs."""

import json
from pathlib import Path
import shutil

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from training.gnn.data import canonical_hash
from data_collection.vertex_family_manifest import read_manifest
from utils.experiment_logger import ExperimentLogger


def selected_frames(path):
    """Return unique, fully checked selection rows that pass both gates."""
    seen = set()
    selected = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate selection sample: {sample_id}")
            seen.add(sample_id)
            if row["both_pass"] != (row["peg_pass"] and row["hole_pass"]):
                raise ValueError(f"Inconsistent physical selection: {sample_id}")
            if row["both_pass"]:
                selected.append(row)
    if not selected:
        raise ValueError("Physical selection contains no frames passing both conditions")
    return selected


def copy_checked(source, destination, expected):
    """Preserve source graph bytes after verifying their recorded checksum."""
    if file_sha256(source) != expected:
        raise ValueError(f"Graph or label checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_sha256(destination) != expected:
        raise ValueError(f"Graph or label copy checksum mismatch: {destination}")


def prepare(config):
    """Copy selected cached graphs and unchanged validation graphs into a new dataset."""
    cache = Path(config["cache_root"]).resolve()
    validation_root = Path(config["validation_dataset"]).resolve()
    validation = read_manifest(validation_root)
    selected = selected_frames(config["selection_frames"])
    inventory_path = cache / "metadata/cache_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    entries = {row["sample_id"]: row for row in inventory["files"]}
    if len(entries) != len(inventory["files"]):
        raise ValueError("Graph inventory contains duplicate sample IDs")
    validation_rows = [row for row in validation["samples"] if row["split"] == "validation"]
    validation_parents = {row["parent_id"] for row in validation_rows}
    if validation_parents & {row["parent_id"] for row in selected}:
        raise ValueError("Selected train and development validation parents overlap")
    run_config = dict(config)
    with ExperimentLogger(team="graph", config=run_config, seed=0,
                          model="selected_vertex_dataset", job_type="data_preparation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        for index, selected_row in enumerate(selected):
            sample_id = selected_row["sample_id"]
            entry = entries[sample_id]
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Cache graph paths must be relative")
            source = (cache / relative).resolve()
            if not source.is_relative_to(cache):
                raise ValueError("Cache graph paths must stay inside cache root")
            graph_rel = Path("graphs") / f"{canonical_hash(sample_id)}.npz"
            label_rel = Path("labels") / f"{canonical_hash(sample_id)}.npz"
            copy_checked(source, output / graph_rel, entry["sha256"])
            with np.load(source, allow_pickle=False) as archive:
                identity = json.loads(str(archive["identity"].item()))
                if identity["sample_id"] != sample_id or identity["frame_sha256"] != selected_row["sha256"]:
                    raise ValueError(f"Cached graph source identity mismatch: {sample_id}")
                valid = archive["node_valid"][0]
                xyz = archive["xyz_world"][0, valid]
            # M0 consumes no family token; unknown placeholders contain no GT labels.
            (output / label_rel).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / label_rel, xyz_world=xyz,
                                detector_family=np.full(len(xyz), -1, dtype=np.int64),
                                oracle_family=np.full(len(xyz), -1, dtype=np.int64))
            rows.append(dict(sample_id=sample_id, parent_id=selected_row["parent_id"], split="train",
                             source_sha256=selected_row["sha256"], sample_identity=identity,
                             graph_path=str(graph_rel), graph_sha256=entry["sha256"],
                             labels_path=str(label_rel), labels_sha256=file_sha256(output / label_rel)))
            if (index + 1) % 1000 == 0:
                print(f"Selected graphs prepared: {index + 1}/{len(selected)}", flush=True)
        for row in validation_rows:
            for key, digest in (("graph_path", "graph_sha256"), ("labels_path", "labels_sha256")):
                copy_checked(validation_root / row[key], output / row[key], row[digest])
            rows.append(row)
        manifest = dict(schema_version=1, original_pool="train_only", reserved_test_frames_read=0,
                        samples=rows, splits={"train": sorted({r["parent_id"] for r in selected}),
                                              "validation": sorted(validation_parents)},
                        source_dataset_sha256=validation["source_dataset_sha256"],
                        selection_frames_sha256=file_sha256(config["selection_frames"]),
                        cache_inventory_sha256=file_sha256(inventory_path),
                        validation_manifest_sha256=file_sha256(validation_root / "manifest.json"),
                        selection_rule="both_pass", student_view="original_graph_m0_no_semantic_ids")
        manifest["dataset_sha256"] = canonical_hash(manifest)
        write_json(output / "manifest.json", manifest)
        write_json(output / "effective_config.json", config)
        logger.record(dataset_sha256=manifest["dataset_sha256"], train_frames=len(selected),
                      validation_frames=len(validation_rows))
        logger.link_output(output / "manifest.json")
        print(f"Selected M0 dataset: {output}", flush=True)
    return output
