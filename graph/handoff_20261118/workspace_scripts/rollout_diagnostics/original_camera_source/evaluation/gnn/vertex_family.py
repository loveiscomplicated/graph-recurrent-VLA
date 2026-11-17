"""Paired development evaluation of vertex-family information and interventions."""

from dataclasses import fields, replace
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_metrics import (
    cluster_interval, per_frame_errors, statistics,
)
from graph.build_graph import GraphBatch
from training.gnn.data import canonical_hash, collate_graph_samples
from training.gnn.tasks import Object21Task
from training.gnn.trainer import build_model
from training.gnn.vertex_family_data import VertexFamilyDataset
from data_collection.vertex_family_manifest import read_manifest
from utils.experiment_logger import ExperimentLogger


METRICS = (
    "peg_position_error_mm", "peg_rotation_error_deg",
    "hole_position_error_mm", "hole_rotation_error_deg",
    "relative_3d_error_mm", "endpoint_position_error_mm",
    "lateral_error_mm", "axial_error_mm",
    "peg_half_size_x_error_mm", "peg_half_size_y_error_mm",
    "peg_half_size_z_error_mm", "hole_radius_error_mm",
)
MODES = ("baseline", "detector", "oracle", "type_control")


def annotation_groups(root, manifest):
    """Separate detector-defined groups from GT-certified analysis-only groups."""
    rows = [row for row in manifest["samples"] if row["split"] == "validation"]
    features = {key: [] for key in ("target_count", "detector_exterior", "detector_rim", "detector_unknown",
                                    "oracle_exterior", "oracle_rim", "oracle_unknown", "oracle_unique_rim",
                                    "wrong_certified", "half_length_mm", "hole_radius_mm")}
    confusion = np.zeros((2, 3), dtype=int)
    by_method = {}
    node_parents, correct, certified_flags, all_node_parents = [], [], [], []
    certified_actual, certified_predicted = [], []
    for row in rows:
        with np.load(Path(root) / row["labels_path"], allow_pickle=False) as archive:
            target = archive["object_type"] == 1
            detected = archive["detector_family"][target]
            oracle = archive["oracle_family"][target]
            physical = archive["physical_id"][target]
            known = oracle >= 0
            features["target_count"].append(int(target.sum()))
            for prefix, labels in (("detector", detected), ("oracle", oracle)):
                for family, name in ((0, "exterior"), (1, "rim"), (-1, "unknown")):
                    features[f"{prefix}_{name}"].append(int((labels == family).sum()))
            features["oracle_unique_rim"].append(len(set(physical[oracle == 1])))
            features["wrong_certified"].append(int(((detected != oracle) & (detected >= 0) & known).sum()))
            features["half_length_mm"].append(float(archive["peg_half_sizes"].reshape(3)[0]) * 1000)
            features["hole_radius_mm"].append(float(archive["hole_radius"].reshape(-1)[0]) * 1000)
            for actual, predicted in zip(oracle[known], detected[known]):
                confusion[actual, predicted if predicted >= 0 else 2] += 1
            metadata = json.loads(str(archive["metadata_json"]))
            for node in np.flatnonzero(target):
                method = metadata[node].get("method", "unknown")
                counts = by_method.setdefault(method, dict(target_nodes=0, certified=0, correct=0,
                                                          certified_exterior=0, certified_rim=0))
                counts["target_nodes"] += 1
                actual = int(archive["oracle_family"][node])
                if actual >= 0:
                    counts["certified"] += 1
                    counts["correct"] += int(archive["detector_family"][node] == actual)
                    counts["certified_exterior" if actual == 0 else "certified_rim"] += 1
            correct.extend((detected[known] == oracle[known]).astype(float))
            certified_actual.extend(oracle[known])
            certified_predicted.extend(detected[known])
            node_parents.extend([row["parent_id"]] * int(known.sum()))
            certified_flags.extend(known.astype(float))
            all_node_parents.extend([row["parent_id"]] * len(detected))
    features = {key: np.asarray(value) for key, value in features.items()}
    count = features["target_count"]
    groups = dict(all=np.ones(len(rows), dtype=bool), missing_target=count == 0,
                  target_present=count > 0, fully_certified_target=(count > 0) & (features["oracle_unknown"] == 0),
                  detector_uncertain=features["detector_unknown"] > 0,
                  oracle_uncertain=features["oracle_unknown"] > 0,
                  detector_wrong_certified=features["wrong_certified"] > 0,
                  oracle_unique_rim_ge3=features["oracle_unique_rim"] >= 3)
    groups["fully_certified_and_detector_known"] = (groups["fully_certified_target"]
                                                      & (features["detector_unknown"] == 0))
    for prefix in ("detector", "oracle"):
        exterior = features[f"{prefix}_exterior"]
        rim = features[f"{prefix}_rim"]
        groups.update({f"{prefix}_exterior_only": (exterior > 0) & (rim == 0),
                       f"{prefix}_rim_only": (rim > 0) & (exterior == 0),
                       f"{prefix}_mixed": (rim > 0) & (exterior > 0),
                       f"{prefix}_rim_1_2": (rim >= 1) & (rim <= 2), f"{prefix}_rim_ge3": rim >= 3})
    for name in ("oracle_exterior_only", "oracle_mixed", "oracle_rim_ge3"):
        groups[f"fully_certified_{name}"] = groups["fully_certified_target"] & groups[name]
    for lower, upper in ((85, 95), (95, 105), (105, 115), (115, 125.001)):
        groups[f"half_length_{lower:g}_{upper:g}"] = (features["half_length_mm"] >= lower) & (features["half_length_mm"] < upper) & (count > 0)
    quality = dict(confusion_gt_rows_exterior_rim_detector_columns_exterior_rim_unknown=confusion.tolist(),
                   certified_accuracy=cluster_interval(np.asarray(correct), np.asarray(node_parents), 2000),
                   certified_node_fraction=cluster_interval(np.asarray(certified_flags), np.asarray(all_node_parents), 2000),
                   target_nodes=int(count.sum()), certified_nodes=int(len(correct)),
                   uncertified_nodes=int((features["oracle_unknown"]).sum()),
                   detector_unknown_nodes=int(features["detector_unknown"].sum()))
    quality["retained_proposal_methods"] = by_method
    if quality["certified_accuracy"] is not None:
        quality["certified_accuracy"]["estimand"] = "certified-node-weighted accuracy"
    if quality["certified_node_fraction"] is not None:
        quality["certified_node_fraction"]["estimand"] = "target-node-weighted certification coverage"
    actual = np.asarray(certified_actual)
    predicted = np.asarray(certified_predicted)
    node_parents = np.asarray(node_parents)
    quality["family_recall_on_certified_nodes"] = {}
    for label, name in ((0, "exterior"), (1, "rim")):
        selected = actual == label
        interval = cluster_interval((predicted[selected] == label).astype(float), node_parents[selected], 2000)
        if interval is not None:
            interval["estimand"] = "certified-node-weighted family recall"
        quality["family_recall_on_certified_nodes"][name] = interval
    return rows, features, groups, quality


def check_graph_parity(datasets):
    """Verify every paired frame's target and all non-semantic graph fields."""
    baseline = datasets["baseline"]
    for mode, dataset in datasets.items():
        if len(dataset) != len(baseline):
            raise ValueError(f"Different validation membership for {mode}")
        for original, changed in zip(baseline.samples, dataset.samples):
            if original.sample_ids != changed.sample_ids:
                raise ValueError("Family modes must preserve sample order and membership")
            for field in fields(GraphBatch):
                if field.name not in ("semantic_id", "semantic_valid"):
                    if not torch.equal(getattr(original.graph, field.name), getattr(changed.graph, field.name)):
                        raise ValueError(f"Family intervention changed geometry field {field.name}")
            if not torch.equal(original.targets["object21"], changed.targets["object21"]):
                raise ValueError("Family intervention changed Object21 supervision")


def semantic_intervention(graph, intervention):
    """Change labels only, preserving UNKNOWN and all observed graph geometry."""
    if intervention == "natural":
        return graph
    semantics = graph.semantic_id.clone()
    valid = graph.semantic_valid.clone()
    if intervention == "erase":
        semantics.fill_(-1)
        valid.fill_(False)
    elif intervention == "swap":
        semantics[valid] = 1 - semantics[valid]
    else:
        raise ValueError(f"Unknown semantic intervention: {intervention}")
    return replace(graph, semantic_id=semantics, semantic_valid=valid)


def infer(model, dataset, device, batch_size, intervention):
    """Evaluate a frozen model, retaining actual target anchor and all Object21 fields."""
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            batch = collate_graph_samples(dataset.samples[start:start + batch_size])
            graph = semantic_intervention(batch.graph, intervention).to(device)
            result = model(graph)
            chunks.append({key: result[key].cpu() for key in ("object21", "endpoint_geom", "endpoint_residual", "hole_anchor")})
    tensors = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
    target = torch.cat([sample.targets["object21"] for sample in dataset.samples])
    errors = per_frame_errors(tensors, target)
    return {key: value.numpy() for key, value in tensors.items()}, target.numpy(), errors


def parameter_usage(model, dataset, device):
    """Count parameters with nonzero gradient in a fixed TRAIN-only diagnostic.

    No optimizer step occurs. Counts describe observed functional use on these
    batches, not statistical degrees of freedom or information content.
    """
    model.eval()
    task = Object21Task()
    used = {name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in model.named_parameters()}
    for start in range(0, min(len(dataset), 128), 32):
        batch = collate_graph_samples(dataset.samples[start:start + 32]).to(device)
        model.zero_grad(set_to_none=True)
        task.compute(model, batch).loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                used[name] |= parameter.grad != 0
    model.zero_grad(set_to_none=True)
    return dict(total=sum(parameter.numel() for parameter in model.parameters()),
                nonzero_gradient_union=sum(int(mask.sum()) for mask in used.values()),
                probe_train_frames=min(len(dataset), 128),
                semantic_rows_used=[index for index, row in enumerate(used["semantic_embed.weight"]) if row.any()],
                nonzero_by_tensor={name: int(mask.sum()) for name, mask in used.items()})


def summarize_metric(values, parents, mask):
    """Report physical quantiles and Parent-clustered mean uncertainty."""
    return {**statistics(values[mask]), "parent_mean_ci95": cluster_interval(values[mask], parents[mask], 2000)}



def seed_metric_arrays(collected, mode, seeds, metric, intervention="natural"):
    """Keep the training-seed axis separate from the paired validation-frame axis."""
    return np.stack([collected[(mode, seed, intervention)][metric] for seed in seeds])


def metric_table_row(group, mode, seed, metric, result, parents, mask):
    """Flatten one explicitly named aggregate into a portable report table row."""
    interval = result["parent_mean_ci95"] or {}
    return dict(group=group, frames=int(mask.sum()), parents=len(set(parents[mask])),
                mode=mode, seed=seed, metric=metric,
                **{name: result[name] for name in ("mean", "median", "p90", "p95")},
                parent_ci_low=interval.get("low"), parent_ci_high=interval.get("high"))


def summarize_groups(collected, seeds, groups, parents):
    """Report each seed and fixed-seed averages without counting seeds as new frames."""
    results, table_rows = {}, []
    comparisons = (("baseline", "detector"), ("baseline", "oracle"),
                   ("baseline", "type_control"), ("detector", "oracle"),
                   ("type_control", "detector"), ("type_control", "oracle"))
    for group, mask in groups.items():
        group_result = dict(frames=int(mask.sum()), parents=len(set(parents[mask])), modes={}, paired={})
        for mode in MODES:
            mode_result = {"per_seed": {str(seed): {} for seed in seeds}}
            for metric in METRICS:
                arrays = seed_metric_arrays(collected, mode, seeds, metric)
                mode_result[metric] = summarize_metric(arrays.mean(axis=0), parents, mask)
                table_rows.append(metric_table_row(group, mode, "mean_errors", metric,
                                                   mode_result[metric], parents, mask))
                for seed, values in zip(seeds, arrays):
                    result = summarize_metric(values, parents, mask)
                    mode_result["per_seed"][str(seed)][metric] = result
                    table_rows.append(metric_table_row(group, mode, seed, metric, result, parents, mask))
            group_result["modes"][mode] = mode_result
        for reference, candidate in comparisons:
            paired_result = {}
            for metric in METRICS:
                reference_values = seed_metric_arrays(collected, reference, seeds, metric)
                candidate_values = seed_metric_arrays(collected, candidate, seeds, metric)
                deltas = reference_values - candidate_values
                per_seed = {str(seed): summarize_metric(delta, parents, mask)
                            for seed, delta in zip(seeds, deltas)}
                paired_result[metric] = dict(seed_average=summarize_metric(deltas.mean(axis=0), parents, mask),
                                              per_seed=per_seed)
            group_result["paired"][f"{reference}_minus_{candidate}"] = paired_result
        results[group] = group_result
    return results, table_rows


def summarize_interventions(collected, seeds, groups, parents):
    """Measure label-removal/swapping sensitivity on fixed model weights and frames."""
    result = {}
    for mode in ("detector", "oracle"):
        result[mode] = {}
        natural = seed_metric_arrays(collected, mode, seeds, "hole_position_error_mm")
        for intervention in ("erase", "swap"):
            changed = seed_metric_arrays(collected, mode, seeds, "hole_position_error_mm", intervention)
            values = changed.mean(axis=0)
            delta = (changed - natural).mean(axis=0)
            result[mode][intervention] = {}
            for group, mask in groups.items():
                result[mode][intervention][group] = dict(
                    hole_position=summarize_metric(values, parents, mask),
                    intervention_minus_natural=summarize_metric(delta, parents, mask))
    return result


def load_training_runs(config, dataset_sha256):
    """Join completed seed partitions and reject duplicates or protocol drift."""
    directories = config.get("training_matrices")
    if directories is None:
        directories = [config["training_matrix"]]
    runs, matrix_hashes = [], {}
    expected_protocol = None
    seen_pairs = set()
    initializations = {}
    for directory in directories:
        matrix_path = Path(directory) / "matrix.json"
        matrix = json.loads(matrix_path.read_text())
        if matrix["core_dataset_sha256"] != dataset_sha256:
            raise ValueError("Training matrix and evaluation dataset hashes differ")
        matrix_hashes[str(matrix_path)] = file_sha256(matrix_path)
        for run in matrix["runs"]:
            pair = (run["mode"], run["seed"])
            if pair in seen_pairs:
                raise ValueError(f"Duplicate training mode/seed: {pair}")
            seen_pairs.add(pair)
            checkpoint = Path(run["checkpoint"])
            if file_sha256(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError(f"Training checkpoint hash changed: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            settings = payload["config"]
            expected_data_hash = canonical_hash([dataset_sha256, run["mode"]])
            if payload["dataset_sha256"] != expected_data_hash or settings["data"]["family_mode"] != run["mode"]:
                raise ValueError("Checkpoint was trained with a different dataset or family view")
            if settings["model"]["use_semantic_ids"] != (run["mode"] != "baseline"):
                raise ValueError("Checkpoint semantic-feature flag disagrees with its experiment mode")
            protocol = {name: settings[name] for name in
                        ("model", "task", "training", "optimizer", "scheduler", "graph", "confidence")}
            protocol["model"] = dict(protocol["model"], use_semantic_ids=False)
            signature = canonical_hash(protocol)
            if expected_protocol is not None and signature != expected_protocol:
                raise ValueError("Family runs differ in a setting other than semantic mode/seed")
            expected_protocol = signature
            maximum_epochs = settings["training"]["epochs"]
            if run["epochs"] != maximum_epochs:
                if not settings["training"].get("early_stopping") or not run.get("early_stopped"):
                    raise ValueError("Training matrix contains an incomplete epoch budget")
            seed = run["seed"]
            initial_hash = run["initialization_sha256"]
            if seed in initializations and initial_hash != initializations[seed]:
                raise ValueError("Family modes did not start from identical tensors")
            initializations[seed] = initial_hash
            runs.append(run)
    budgets = {(run["epochs"], run["updates"]) for run in runs}
    if len(budgets) != 1 and not all(run.get("early_stop_enabled", False) for run in runs):
        raise ValueError("Family runs have different epoch/update budgets without early stopping")
    runs.sort(key=lambda run: (run["seed"], MODES.index(run["mode"])))
    return runs, matrix_hashes


def evaluate(config):
    """Score all seeds and label-only interventions on the same development frames."""
    torch.set_num_threads(config.get("torch_threads", 4))
    root = Path(config["family_dataset"])
    manifest = read_manifest(root)
    training_runs, matrix_hashes = load_training_runs(config, manifest["dataset_sha256"])
    rows, features, groups, quality = annotation_groups(root, manifest)
    parents = np.asarray([row["parent_id"] for row in rows])
    identifiers = np.asarray([row["sample_id"] for row in rows])
    datasets = {mode: VertexFamilyDataset(root, manifest, "validation", mode) for mode in MODES}
    check_graph_parity(datasets)
    device = torch.device(config["device"])
    collected = {}
    usages = {}
    run_rows = []
    train_reference = None
    with ExperimentLogger(team="graph", config=config, seed=0, model="vertex_family_comparison", job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        for run in training_runs:
            checkpoint = Path(run["checkpoint"])
            if file_sha256(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed after training: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_model(payload["config"]["model"], device)
            model.load_state_dict(payload["model"], strict=True)
            mode, seed = run["mode"], run["seed"]
            if seed == 0:
                train_dataset = VertexFamilyDataset(root, manifest, "train", mode)
                if mode == "baseline":
                    train_reference = train_dataset
                if train_reference is None:
                    raise ValueError("Baseline seed-zero training view is required for full graph parity checks")
                check_graph_parity({"baseline": train_reference, mode: train_dataset})
                usages[mode] = parameter_usage(model, train_dataset, device)
            interventions = ("natural", "erase", "swap") if mode in ("detector", "oracle") else ("natural",)
            for intervention in interventions:
                arrays, target, errors = infer(model, datasets[mode], device, config.get("batch_size", 128), intervention)
                key = f"{mode}_seed_{seed}_{intervention}"
                np.savez_compressed(output / f"{key}.npz", sample_ids=identifiers, target=target, **arrays, **errors)
                collected[(mode, seed, intervention)] = errors
                if intervention == "natural":
                    desired_offset = target[:, 10:13] - arrays["hole_anchor"]
                    prediction_offset = arrays["object21"][:, 10:13] - arrays["hole_anchor"]
                    arrays["offset_error_mm"] = (prediction_offset - desired_offset) * 1000
                    np.savez_compressed(output / f"{mode}_seed_{seed}_head_diagnostics.npz",
                                        sample_ids=identifiers, desired_offset=desired_offset,
                                        predicted_offset=prediction_offset, signed_error_mm=arrays["offset_error_mm"])
                print(f"Family evaluation complete: {key}", flush=True)
            run_rows.append({**run, "parameters": sum(parameter.numel() for parameter in model.parameters())})
            logger.link_output(checkpoint, checkpoint=True)
        seeds = sorted({run["seed"] for run in training_runs})
        if any((mode, seed, "natural") not in collected for mode in MODES for seed in seeds):
            raise ValueError("Evaluation requires all four modes for every training seed")
        group_summaries, table_rows = summarize_groups(collected, seeds, groups, parents)
        summary = dict(groups=group_summaries, label_quality=quality, parameter_usage=usages,
                       seeds=seeds, runs=run_rows, core_dataset_sha256=manifest["dataset_sha256"],
                       frame_parity_verified=True)
        interventions_summary = summarize_interventions(collected, seeds, groups, parents)
        write_json(output / "summary.json", summary)
        write_json(output / "matrix.json", dict(core_dataset_sha256=manifest["dataset_sha256"], runs=run_rows,
                                                source_matrix_sha256=matrix_hashes))
        write_json(output / "interventions.json", interventions_summary)
        write_json(output / "effective_config.json", config)
        np.savez_compressed(output / "observation_groups.npz", sample_ids=identifiers, parent_ids=parents, **features, **groups)
        with (output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
        source_paths = [Path(__file__), Path("graph/gated_mpnn.py"), Path("graph/vertex_family.py"),
                        Path("training/gnn/vertex_family.py"), Path("training/gnn/vertex_family_data.py"),
                        Path("data_collection/vertex_family.py"), Path("data_collection/vertex_family_labels.py")]
        write_json(output / "provenance.json", dict(dataset_sha256=manifest["dataset_sha256"],
                   manifest_sha256=file_sha256(root / "manifest.json"), matrix_sha256=matrix_hashes,
                   source_sha256={str(path): file_sha256(path) for path in source_paths},
                   checkpoint_sha256={f"{run['mode']}/seed_{run['seed']}": run["checkpoint_sha256"] for run in run_rows},
                   torch_version=torch.__version__, cuda_version=torch.version.cuda, device=str(device),
                   reserved_test_frames_read=0, original_validation_frames_read=0))
        logger.record(core_dataset_sha256=manifest["dataset_sha256"], development_validation_frames=len(rows),
                      development_validation_parents=len(set(parents)), reserved_test_frames_read=0)
        for path in output.iterdir():
            if path.is_file():
                logger.link_output(path)
        print(f"Vertex family paired evaluation: {output}", flush=True)
    return output
