"""Observed-coordinate coverage, train priors and fixed-input perturbation probes."""

from copy import deepcopy
from dataclasses import fields
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_diagnostics import MODELS, group_masks, selected_cases, size_diagnostics
from evaluation.gnn.vertex_metrics import (
    load_aligned_predictions, statistics,
)
from graph.build_graph import GraphBatch, GraphBuilder, GraphSettings
from graph.pose_geometry import conjugate, rotate_vector
from training.gnn.cached_data import build_data
from training.gnn.data import collate_graph_samples, TrainingBatch
from training.gnn.trainer import build_model
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def geometry_coverage(geometry, predictions):
    """Measure GT-axis coverage as a diagnostic; coordinates remain raw model inputs."""
    baseline = predictions["baseline"]
    lookup = {identifier: index for index, identifier in enumerate(geometry["sample_ids"])}
    order = [lookup[identifier] for identifier in baseline["sample_ids"]]
    target = baseline["target"].astype(np.float64)
    np.testing.assert_array_equal(geometry["target"][order], baseline["target"])
    xyz = geometry["xyz_world"][order].astype(np.float64)
    labels = geometry["object_type"][order]
    tcp = geometry["tcp_world"][order].astype(np.float64)
    displacement = xyz - tcp[:, None, :] - target[:, None, :3]
    local = rotate_vector(conjugate(torch.tensor(target[:, None, 3:7])), torch.tensor(displacement)).numpy()
    ratios, half_spans, near_corners, endpoint_sides = [], [], [], []
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    case_lookup = {case["index"]: case for case in selected_cases(predictions)}
    for index, points in enumerate(local):
        peg = points[labels[index] == 0]
        half_size = target[index, 17:20]
        span = np.ptp(peg[:, 0]) if len(peg) > 1 else 0
        half_spans.append(span * 500)
        ratios.append(span / (2 * half_size[0]))
        both_ends = bool((peg[:, 0] <= -.8 * half_size[0]).any() and (peg[:, 0] >= .8 * half_size[0]).any())
        endpoint_sides.append(both_ends)
        corners = signs * half_size
        distances = np.linalg.norm(peg[:, None, :] - corners[None, :, :], axis=2)
        closest = distances.argmin(1)
        within = distances.min(1) <= .005
        near_corners.append(len(set(closest[within])))
        if index in case_lookup:
            case_lookup[index].update(
                observed_xyz_world_m=xyz[index][labels[index] >= 0].tolist(),
                observed_object_types=labels[index][labels[index] >= 0].tolist(),
                peg_xyz_gt_local_m=peg.tolist(), axial_span_fraction=float(ratios[-1]),
                both_end_bands=both_ends, unique_gt_corners_within_5mm=near_corners[-1],
                tcp_world_m=tcp[index].tolist(),
            )
    ratios = np.asarray(ratios)
    half_spans = np.asarray(half_spans)
    near_corners = np.asarray(near_corners)
    endpoint_sides = np.asarray(endpoint_sides)
    summaries = {}
    for group, mask in group_masks(baseline["counts"]).items():
        summaries[group] = {
            "frames": int(mask.sum()), "axial_span_fraction": statistics(ratios[mask]),
            "both_end_bands_fraction": float(endpoint_sides[mask].mean()) if mask.any() else None,
            "unique_gt_corners_within_5mm": statistics(near_corners[mask]),
            "gt_axis_span_half_size_error_mm": statistics(np.abs(half_spans - target[:, 17] * 1000)[mask]),
        }
    return summaries, list(case_lookup.values()), dict(
        axial_span_fraction=ratios, axial_half_span_mm=half_spans,
        unique_gt_corners_within_5mm=near_corners, both_end_bands=endpoint_sides,
    )


def constant_predictors(train_geometry, predictions):
    """Fit constants exclusively on the complete training split and score validation."""
    train_size = train_geometry["target"][:, 17].astype(np.float64) * 1000
    target = predictions["baseline"]["target"][:, 17].astype(np.float64) * 1000
    result = {
        "train_frames": len(train_size), "validation_frames": len(target),
        "train_size_x_mm": {**statistics(train_size), "std": float(train_size.std()),
                             "min": float(train_size.min()), "max": float(train_size.max())},
        "validation_size_x_mm": {**statistics(target), "std": float(target.std()),
                                  "min": float(target.min()), "max": float(target.max())},
        "constants": {},
    }
    for name, value in (("train_mean", train_size.mean()), ("train_median", np.median(train_size))):
        error = np.abs(target - value)
        result["constants"][name] = {
            "size_x_mm": float(value),
            "groups": {group: statistics(error[mask]) for group, mask in
                       group_masks(predictions["baseline"]["counts"]).items()},
        }
    return result


def case_attention(checkpoints, cases, device="cpu"):
    """Inspect original readout attention; attention mass is not causal attribution."""
    device = set_device(device)
    result = {}
    for name in MODELS:
        payload = torch.load(checkpoints[name], map_location="cpu", weights_only=False)
        settings = payload["config"]["model"]
        if settings["use_semantic_ids"] or settings["use_confidence"]:
            raise ValueError("Coordinate-only case replay requires disabled semantics/confidence")
        model = build_model(settings, torch.device(device)).eval()
        model.load_state_dict(payload["model"], strict=True)
        builder = GraphBuilder(GraphSettings(**payload["config"]["graph"]))
        captured = {}

        def capture(module, arguments):
            captured["inputs"] = arguments

        handle = model.peg_pool.register_forward_pre_hook(capture)
        try:
            result[name] = []
            with torch.no_grad():
                for case in cases:
                    xyz = torch.tensor(case["observed_xyz_world_m"], dtype=torch.float32)
                    labels = torch.tensor(case["observed_object_types"], dtype=torch.long)
                    observation = SimpleNamespace(xyz=xyz, object_id=(labels != 0).long(),
                                                  metadata=[{} for _ in labels])
                    graph = builder([observation], torch.tensor([case["tcp_world_m"]]),
                                    object_types=[labels]).to(device)
                    output = model(graph)
                    predicted = float(output["object21"][0, 17] * 1000)
                    np.testing.assert_allclose(predicted, case["predicted_size_x_mm"][name], atol=.01)
                    h, confidence, valid = captured["inputs"]
                    pool = model.peg_pool
                    if hasattr(pool, "attention"):
                        logits = pool.attention(h).squeeze(-1) + confidence.clamp_min(1e-6).log()
                        weights = logits.masked_fill(~valid, -torch.inf).softmax(-1)[0, valid[0]]
                        weights_by_seed = weights[None, None, :]
                    else:
                        query = pool.query(pool.seeds).reshape(-1, pool.heads, pool.head_dim)
                        key = pool.key(h).reshape(1, h.shape[1], pool.heads, pool.head_dim)
                        logits = torch.einsum("shd,bnhd->bhsn", query, key) / math.sqrt(pool.head_dim)
                        logits += confidence.clamp_min(1e-6).log()[:, None, None, :]
                        weights_by_seed = logits.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)[0, :, :, valid[0]]
                        weights = weights_by_seed.mean((0, 1))
                    local_x = np.asarray(case["peg_xyz_gt_local_m"])[:, 0]
                    negative = torch.tensor(local_x < 0, device=weights.device)
                    result[name].append(dict(
                        sample_id=case["sample_id"], predicted_size_x_mm=predicted,
                        peg_local_x_mm=(local_x * 1000).tolist(),
                        mean_node_attention=weights.cpu().tolist(),
                        attention_heads_seeds_nodes=weights_by_seed.cpu().tolist(),
                        negative_end_attention_mass=float(weights[negative].sum()),
                    ))
        finally:
            handle.remove()
    return result


def perturb_graph(graph, condition, seed):
    """Change only PEG observations and their derived edges, keeping TCP fixed."""
    graph = GraphBatch(**{field.name: getattr(graph, field.name).clone() for field in fields(graph)})
    peg = graph.node_valid & (graph.object_type == 0)
    generator = torch.Generator().manual_seed(seed)
    if condition.startswith("drop"):
        fraction = .25 if condition == "drop25" else .5
        indices = torch.nonzero(peg[0], as_tuple=True)[0]
        removed_count = max(1, int(len(indices) * fraction))
        removed = indices[torch.randperm(len(indices), generator=generator)[:removed_count]]
        graph.node_valid[0, removed] = False
        graph.object_type[0, removed] = -1
        graph.xyz_world[0, removed] = 0
        graph.semantic_id[0, removed] = -1
        graph.semantic_valid[0, removed] = False
        graph.confidence[0, removed] = 0
        graph.confidence_valid[0, removed] = False
        keep_edges = graph.node_valid.flatten()[graph.edge_index].all(0)
        graph.edge_index = graph.edge_index[:, keep_edges]
        graph.edge_type = graph.edge_type[keep_edges]
    elif condition.startswith("noise"):
        sigma = .0005 if condition == "noise0.5mm" else .001
        noise = torch.randn(graph.xyz_world.shape, generator=generator) * sigma
        graph.xyz_world += noise * peg.unsqueeze(-1)
    elif condition != "original":
        raise ValueError(f"Unknown perturbation: {condition}")
    source, destination = graph.edge_index
    flat_xyz = graph.xyz_world.reshape(-1, 3)
    graph.relative_xyz = flat_xyz[source] - flat_xyz[destination]
    graph.distance = graph.relative_xyz.norm(dim=-1, keepdim=True)
    graph.validate()
    return graph


def sensitivity(config):
    """Hold weights, frames, perturbation draws and all targets fixed across models."""
    predictions, prediction_hashes = load_aligned_predictions(config["predictions"])
    baseline = predictions["baseline"]
    high = group_masks(baseline["counts"])["high_support_proxy"]
    random = np.random.default_rng(0)
    count = config.get("sensitivity_frames_per_group", 256)
    chosen = np.concatenate([random.choice(np.flatnonzero(mask), min(count, mask.sum()), replace=False)
                             for mask in (high, ~high)])
    identifiers = baseline["sample_ids"][chosen]
    payload = torch.load(config["checkpoints"]["baseline"], map_location="cpu", weights_only=False)
    data_config = deepcopy(payload["config"])
    data_config["data"]["cache_dir"] = config["cache_dir"]
    data_config["data"]["num_workers"] = 0
    bundle = build_data(data_config, torch.Generator().manual_seed(0))
    dataset = bundle.validation.dataset
    index = {sample.sample_id: row for row, sample in enumerate(dataset.samples)}
    samples = [dataset[index[identifier]] for identifier in identifiers]
    device = set_device(config.get("device", "cpu"))
    conditions = ("original", "drop25", "drop50", "noise0.5mm", "noise1mm")
    arrays = {}
    for name in MODELS:
        payload = torch.load(config["checkpoints"][name], map_location="cpu", weights_only=False)
        model = build_model(payload["config"]["model"], device).eval()
        model.load_state_dict(payload["model"], strict=True)
        for condition in conditions:
            values = []
            with torch.inference_mode():
                for start in range(0, len(samples), 64):
                    changed = [TrainingBatch(perturb_graph(sample.graph, condition, 1000 + start + offset),
                                             sample.targets, sample.sample_ids)
                               for offset, sample in enumerate(samples[start:start + 64])]
                    batch = collate_graph_samples(changed)
                    output = model(batch.graph.to(device))["object21"].cpu().numpy()
                    values.append(output)
            arrays[f"{name}_{condition}"] = np.concatenate(values)
            if condition == "original":
                np.testing.assert_allclose(arrays[f"{name}_{condition}"], predictions[name]["object21"][chosen],
                                           atol=1e-5, rtol=1e-4)
    result = {}
    target = baseline["target"][chosen]
    for name in MODELS:
        original = arrays[f"{name}_original"][:, 17]
        result[name] = {}
        for condition in conditions:
            predicted = arrays[f"{name}_{condition}"][:, 17]
            result[name][condition] = {}
            for group, mask in (("high_support_proxy", high[chosen]), ("low_support_proxy", ~high[chosen])):
                result[name][condition][group] = {
                    "frames": int(mask.sum()),
                    "size_x_error_mm": statistics(np.abs(predicted - target[:, 17])[mask] * 1000),
                    "absolute_prediction_change_mm": statistics(np.abs(predicted - original)[mask] * 1000),
                }
    return result, dict(sample_ids=identifiers, target=target, high_support=high[chosen], **arrays), prediction_hashes


def analyze(config):
    """Analyze raw validation geometry and complete training priors, then perturb."""
    predictions, hashes = load_aligned_predictions(config["predictions"])
    with np.load(config["train_geometry"], allow_pickle=False) as archive:
        train = dict(archive)
    with np.load(config["validation_geometry"], allow_pickle=False) as archive:
        validation = dict(archive)
    coverage, cases, arrays = geometry_coverage(validation, predictions)
    constants = constant_predictors(train, predictions)
    effective = dict(config, prediction_sha256=hashes,
                     train_geometry_sha256=file_sha256(config["train_geometry"]),
                     validation_geometry_sha256=file_sha256(config["validation_geometry"]))
    with ExperimentLogger(team="graph", config=effective, seed=0,
                          model="vertex_geometry_diagnostics", job_type="evaluate") as experiment:
        output = Path(config["output"]["root"]).resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "coverage.json", coverage)
        write_json(output / "cases.json", cases)
        write_json(output / "constant_predictors.json", constants)
        if config.get("inspect_case_attention", False):
            attention = case_attention(config["checkpoints"], cases, config.get("device", "cpu"))
            write_json(output / "case_attention.json", attention)
        np.savez_compressed(output / "coverage_arrays.npz", sample_ids=predictions["baseline"]["sample_ids"], **arrays)
        if config.get("parent_manifest"):
            parents = json.loads(Path(config["parent_manifest"]).read_text())["parent_by_episode"]
            high = group_masks(predictions["baseline"]["counts"])["high_support_proxy"]
            conditions = {
                "high_both_end_bands": high & arrays["both_end_bands"],
                "high_not_both_end_bands": high & ~arrays["both_end_bands"],
                "high_span_98_102_percent": high & (arrays["axial_span_fraction"] >= .98) &
                    (arrays["axial_span_fraction"] <= 1.02),
            }
            gt_size = predictions["baseline"]["target"][:, 17] * 1000
            for lower, upper in ((85, 95), (95, 105), (105, 115), (115, 125)):
                conditions[f"high_size_{lower}_{upper}mm"] = high & (gt_size >= lower) & (gt_size < upper)
            conditioned = size_diagnostics(predictions, parents, 2000, groups=conditions)
            write_json(output / "coverage_conditioned_errors.json", conditioned)
        if config.get("sensitivity", True):
            result, values, _ = sensitivity(config)
            write_json(output / "sensitivity.json", result)
            np.savez_compressed(output / "sensitivity_predictions.npz", **values)
        write_json(output / "effective_config.json", effective)
        for path in output.iterdir():
            experiment.link_output(path, kind="metrics")
        print(f"Vertex geometry diagnostics: {output}", flush=True)
        return output
