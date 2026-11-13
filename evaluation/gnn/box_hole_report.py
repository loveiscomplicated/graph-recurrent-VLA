"""Export report tables and publication figures from completed, hashed evaluations."""

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.box_hole_diagnostics import diagnostic_groups, load_geometry
from evaluation.gnn.box_hole_plots import make_plots
from evaluation.gnn.vertex_metrics import (
    cluster_interval, statistics,
)
from utils.experiment_logger import ExperimentLogger


def export(config):
    """Preserve original evaluation files while producing auditable report exports."""
    source = Path(config["evaluation"])
    summary_path = source / "summary.json"
    if file_sha256(summary_path) != config["summary_sha256"]:
        raise ValueError("Source evaluation summary differs from the pinned SHA256")
    summary = json.loads(summary_path.read_text())
    states = json.loads((source / "verified_fixture_states.json").read_text())
    geometry = load_geometry(config["validation_geometry"])
    coverage = load_geometry(source / "observation_geometry.npz")
    np.testing.assert_array_equal(coverage.pop("sample_ids"), geometry["sample_ids"])
    groups = diagnostic_groups(geometry["counts"], coverage)
    with (source / "geometric_fitting.csv").open() as stream:
        fitting = []
        for row in csv.DictReader(stream):
            fitting.append({key: value if key in ("episode", "family", "subset") else float(value)
                            for key, value in row.items()})
    inference = {}
    for condition in summary["groups"]["all"]["conditions"]:
        inference[condition] = {}
        for model in summary["groups"]["all"]["conditions"][condition]:
            arrays = load_geometry(source / f"{model}_{condition}.npz")
            error = np.linalg.norm(arrays["object21"][:, 10:13] - geometry["target"][:, 10:13], axis=1) * 1000
            inference[condition][model] = dict(arrays=arrays, errors={"hole_position_error_mm": error})
    with ExperimentLogger(team="graph", config=config, seed=0, model="box_hole_report", job_type="analysis") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(summary_path, output / "summary.json")
        table_rows = []
        for group, row in summary["groups"].items():
            for condition, models in row["conditions"].items():
                for model, metrics in models.items():
                    for metric, values in metrics.items():
                        interval = values["parent_mean_ci95"] or {}
                        gain = row["paired"][condition][model][metric]
                        gain_interval = gain["parent_mean_ci95"] or {}
                        table_rows.append(dict(group=group, frames=row["frames"], parents=row["parents"],
                            condition=condition, model=model, metric=metric,
                            **{key: values[key] for key in ("mean", "median", "p90", "p95")},
                            ci95_low=interval.get("low"), ci95_high=interval.get("high"),
                            original_minus_condition_mean=gain["mean"],
                            paired_ci95_low=gain_interval.get("low"), paired_ci95_high=gain_interval.get("high")))
        with (output / "paired_metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
        fitting_summary = {}
        for family in ("outer", "rim"):
            for subset in ("triangle", "one_face", "full"):
                for sigma in (0, .5, 1):
                    selected = [row for row in fitting if row["family"] == family and row["subset"] == subset and row["sigma_mm"] == sigma]
                    fitting_summary[f"{family}/{subset}/{sigma:g}mm"] = {}
                    for metric in ("hole_error_mm", "rotation_error_deg"):
                        values = np.asarray([row[metric] for row in selected])
                        parents = np.asarray([row["episode"] for row in selected])
                        fitting_summary[f"{family}/{subset}/{sigma:g}mm"][metric] = {
                            **statistics(values), "parent_mean_ci95": cluster_interval(values, parents, 2000)}
        offsets = np.asarray([row["box_center_to_hole_mm"] for row in fitting
                              if row["family"] == "outer" and row["subset"] == "full" and row["sigma_mm"] == 0])
        fitting_summary["gt_box_center_as_hole_parent_weighted_mm"] = statistics(offsets)
        source_identity = json.loads(Path(config["collection_source_identity"]).read_text())
        source_checks = {path: file_sha256(path) == source_identity["files"][path] for path in
                         ("data_collection/scripted_policy.py", "data_collection/collect_episodes.py", "envs/maniskill/peg_insertion.py")}
        if not all(source_checks.values()):
            raise ValueError("Collection/policy source changed; seed replay provenance must be reassessed")
        write_json(output / "fitting_summary.json", fitting_summary)
        write_json(output / "collection_source_checks.json", source_checks)
        make_plots(output, geometry, states, coverage, groups, inference, fitting)
        snapshot = output / "sources"
        files = [Path("main.py"), Path("tests/test_box_hole_diagnostics.py")]
        for directory in ("evaluation/gnn", "graph", "graph/build_graph", "training/gnn", "utils", "envs/maniskill"):
            files.extend(Path(directory).glob("*.py"))
        for path in files:
            destination = snapshot / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        write_json(output / "source_sha256.json", {str(path): file_sha256(path) for path in files})
        write_json(output / "effective_config.json", config)
        logger.record(source_evaluation=str(source), source_evaluation_summary_sha256=config["summary_sha256"])
        for path in output.rglob("*"):
            if path.is_file():
                logger.link_output(path)
        print(f"Report exports saved: {output}", flush=True)
    return str(output)
