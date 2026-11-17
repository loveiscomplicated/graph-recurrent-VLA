"""Paired BOX-to-Hole observability and frozen-checkpoint Oracle evaluation.

Run through main.py with graph/configs/analyze_box_hole.yaml. All outputs are
isolated under a new ExperimentLogger run ID. No detector, training, publishing,
reserved-test frame access or checkpoint writes occur.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.box_hole_geometry import (
    SIGNS, geometric_oracles, physical_vertices, replay_validation, rotation_matrix,
    transform_points,
)
from evaluation.gnn.vertex_metrics import (
    cluster_interval, load_aligned_predictions, per_frame_errors, statistics, correlation,
)
from graph.build_graph import GraphBatch
from training.gnn.trainer import build_model
from utils.experiment_logger import ExperimentLogger


METRICS = ("hole_position_error_mm", "hole_rotation_error_deg", "relative_3d_error_mm",
           "lateral_error_mm", "axial_error_mm")
CONDITIONS = ("original", "gt_outer_original_peg", "gt_rim_original_peg",
              "gt_target_original_peg", "full_gt")


def load_geometry(path):
    """Load a numeric geometry export without pickle or any reserved frames."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def graph_from_arrays(xyz, labels, tcp):
    """Build complete directed edges among present nodes with fresh distances.

    Original checkpoints disable semantic IDs and confidence. Unknown identities
    remain unknown even for the GT-coordinate upper references.
    """
    xyz = torch.as_tensor(xyz, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    tcp = torch.as_tensor(tcp, dtype=torch.float32)
    valid = labels >= 0
    batch_size, width = labels.shape
    pairs = valid[:, :, None] & valid[:, None, :]
    pairs &= ~torch.eye(width, dtype=torch.bool)[None]
    batch, source, destination = torch.where(pairs)
    edge_index = torch.stack((batch * width + source, batch * width + destination))
    edge_type = torch.where(labels[batch, source] == labels[batch, destination], labels[batch, source], 3)
    relative = xyz[batch, source] - xyz[batch, destination]
    graph = GraphBatch(xyz_world=xyz, object_type=labels, node_valid=valid,
                       semantic_id=torch.full_like(labels, -1), semantic_valid=torch.zeros_like(valid),
                       confidence=torch.zeros_like(labels, dtype=torch.float32), confidence_valid=torch.zeros_like(valid),
                       tcp_xyz_world=tcp, edge_index=edge_index, edge_type=edge_type,
                       relative_xyz=relative, distance=relative.norm(dim=-1, keepdim=True))
    graph.validate()
    return graph


def oracle_arrays(geometry, states, condition):
    """Construct explicitly more informative GT graphs, never coordinate correction.

    All target-side physical vertices keep BOX type, matching the detector's
    existing type vocabulary. GT rim coordinates therefore do not add a HOLE
    label or semantic embedding. Visibility is intentionally changed here.
    """
    if condition == "original":
        return geometry["xyz_world"], geometry["object_type"]
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown Oracle condition: {condition}")
    xyz_rows, label_rows = [], []
    for index, identifier in enumerate(geometry["sample_ids"]):
        state = states[str(Path(identifier).parent)]
        outer, rim = physical_vertices(state["box_pose"], state["hole_pose"],
                                       state["peg_half_sizes"][0], state["hole_radius"])
        target = {"gt_outer_original_peg": outer, "gt_rim_original_peg": rim}.get(condition)
        if target is None:
            target = np.concatenate((outer, rim))
        peg = geometry["xyz_world"][index][geometry["object_type"][index] == 0]
        if condition == "full_gt":
            gt = geometry["target"][index]
            pose = np.concatenate((gt[:3] + geometry["tcp_world"][index], gt[3:7]))
            peg = transform_points(SIGNS * gt[17:20], pose)
        xyz_rows.append(np.concatenate((peg, target)))
        label_rows.append(np.concatenate((np.zeros(len(peg), dtype=np.int64), np.ones(len(target), dtype=np.int64))))
    width = max(map(len, xyz_rows))
    xyz = np.zeros((len(xyz_rows), width, 3), dtype=np.float32)
    labels = np.full((len(xyz_rows), width), -1, dtype=np.int64)
    for index, (points, kinds) in enumerate(zip(xyz_rows, label_rows)):
        xyz[index, :len(points)] = points
        labels[index, :len(points)] = kinds
    return xyz, labels


def describe_input(xyz, labels, tcp):
    """Record simple input-distribution measurements without target leakage."""
    result = {key: [] for key in ("target_count", "target_diameter_mm", "target_anchor_norm_mm", "target_min_spacing_mm")}
    for points, kinds, origin in zip(xyz, labels, tcp):
        points = points[kinds > 0]
        result["target_count"].append(len(points))
        result["target_anchor_norm_mm"].append(np.linalg.norm(points.mean(0) - origin) * 1000 if len(points) else 0)
        if len(points) > 1:
            distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
            result["target_diameter_mm"].append(distance.max() * 1000)
            distance[np.diag_indices(len(points))] = np.inf
            result["target_min_spacing_mm"].append(distance.min() * 1000)
        else:
            result["target_diameter_mm"].append(0)
            result["target_min_spacing_mm"].append(0)
    return {key: np.asarray(value) for key, value in result.items()}


def observation_geometry(geometry, states):
    """Measure distribution and GT proximity, explicitly without certifying identity."""
    count = len(geometry["sample_ids"])
    arrays = {key: np.zeros(count) for key in (
        "rank_1mm", "s2_mm", "s3_mm", "diameter_mm", "unique_outer_candidates_3mm",
        "unique_rim_candidates_3mm", "unmatched_nodes_3mm", "ambiguous_nodes_3mm",
        "hole_offset_mm", "half_length_mm", "hole_radius_mm", "yaw_deg")}
    arrays["anchor_world"] = np.zeros((count, 3))
    for index, identifier in enumerate(geometry["sample_ids"]):
        state = states[str(Path(identifier).parent)]
        points = geometry["xyz_world"][index][geometry["object_type"][index] == 1].astype(np.float64)
        outer, rim = physical_vertices(state["box_pose"], state["hole_pose"],
                                       state["peg_half_sizes"][0], state["hole_radius"])
        if len(points):
            arrays["anchor_world"][index] = points.mean(0)
            singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
            singular = np.pad(singular, (0, 3 - len(singular)))
            arrays["s2_mm"][index], arrays["s3_mm"][index] = singular[1:] * 1000
            arrays["rank_1mm"][index] = (singular > .001).sum()
            arrays["diameter_mm"][index] = np.linalg.norm(points[:, None] - points[None], axis=-1).max() * 1000
            distances = np.linalg.norm(points[:, None] - np.concatenate((outer, rim))[None], axis=-1)
            candidates = distances <= .003
            # A unique nearby reference is still a proximity candidate, not a
            # trustworthy semantic correspondence. It never drives substitution.
            unambiguous = candidates.sum(1) == 1
            supported = np.flatnonzero(candidates[unambiguous].any(0))
            arrays["unique_outer_candidates_3mm"][index] = (supported < 8).sum()
            arrays["unique_rim_candidates_3mm"][index] = (supported >= 8).sum()
            arrays["unmatched_nodes_3mm"][index] = (candidates.sum(1) == 0).sum()
            arrays["ambiguous_nodes_3mm"][index] = (candidates.sum(1) > 1).sum()
        arrays["hole_offset_mm"][index] = np.linalg.norm(np.asarray(state["hole_pose"])[:3] - np.asarray(state["box_pose"])[:3]) * 1000
        arrays["half_length_mm"][index] = state["peg_half_sizes"][0] * 1000
        arrays["hole_radius_mm"][index] = state["hole_radius"] * 1000
        arrays["yaw_deg"][index] = Rotation.from_matrix(rotation_matrix(state["box_pose"])).as_euler("xyz", degrees=True)[2]
    return arrays


def diagnostic_groups(counts, coverage):
    """Use predeclared bins; coverage-proxy groups do not establish identifiability."""
    box_count = counts[:, 1]
    groups = dict(all=np.ones(len(counts), dtype=bool), target_present=box_count > 0,
                  box_0=box_count == 0, box_1_3=(box_count > 0) & (box_count <= 3),
                  box_ge4=box_count >= 4, high_support_proxy=(counts[:, 0] >= 4) & (box_count >= 4),
                  proximity_rim_ge3=coverage["unique_rim_candidates_3mm"] >= 3,
                  proximity_outer_ge3_no_rim=(coverage["unique_outer_candidates_3mm"] >= 3) & (coverage["unique_rim_candidates_3mm"] == 0),
                  distribution_rank_le1=(box_count > 0) & (coverage["rank_1mm"] <= 1),
                  distribution_planar=coverage["rank_1mm"] == 2,
                  distribution_3d=coverage["rank_1mm"] == 3)
    for name, values, boundaries in (
        ("half_length", coverage["half_length_mm"], (85, 95, 105, 115, 125.001)),
        ("hole_radius", coverage["hole_radius_mm"], (18, 21.333, 24.667, 28.001)),
        ("yaw", coverage["yaw_deg"], (67.49, 78.75, 90, 101.25, 112.51)),
        ("rim_candidates", coverage["unique_rim_candidates_3mm"], (0, 1, 3, 5, 9)),
    ):
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            groups[f"{name}_{lower:g}_{upper:g}"] = (values >= lower) & (values < upper) & (box_count > 0)
    return groups


def summarize(values, parents, mask, repetitions):
    """Return frame statistics plus a whole-parent bootstrap CI for the mean."""
    return {**statistics(values[mask]), "parent_mean_ci95": cluster_interval(values[mask], parents[mask], repetitions)}


def evaluate_models(config, geometry, states, predictions, output):
    """Freeze four original checkpoints; replay original inputs before GT Oracles."""
    device = torch.device(config.get("device", "cpu"))
    inference = {}
    distributions = {}
    replay_checks = {}
    for condition in CONDITIONS:
        xyz, labels = oracle_arrays(geometry, states, condition)
        distributions[condition] = describe_input(xyz, labels, geometry["tcp_world"])
        inference[condition] = {}
        for name, path in config["checkpoints"].items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            settings = payload["config"]["model"]
            if settings["use_semantic_ids"] or settings["use_confidence"]:
                raise ValueError("Geometry-only replay requires semantic/confidence inputs disabled")
            if any(value is not None for value in payload["config"]["graph"].values()):
                raise ValueError("This replay supports the source checkpoints' complete graph only")
            model = build_model(settings, device).eval().requires_grad_(False)
            model.load_state_dict(payload["model"], strict=True)
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(xyz), config.get("batch_size", 128)):
                    stop = start + config.get("batch_size", 128)
                    graph = graph_from_arrays(xyz[start:stop], labels[start:stop], geometry["tcp_world"][start:stop]).to(device)
                    result = model(graph)
                    chunks.append({key: result[key].cpu().numpy() for key in
                                   ("object21", "endpoint_geom", "endpoint_residual", "hole_anchor")})
            arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
            np.savez_compressed(output / f"{name}_{condition}.npz", sample_ids=geometry["sample_ids"], **arrays)
            if condition == "original":
                delta = float(np.max(np.abs(arrays["object21"] - predictions[name]["object21"])))
                np.testing.assert_allclose(arrays["object21"], predictions[name]["object21"], atol=1e-5, rtol=1e-4)
                anchor = distributions[condition]["target_count"] > 0
                expected = geometry["xyz_world"].astype(np.float64)
                kinds = geometry["object_type"] > 0
                expected = (expected * kinds[..., None]).sum(1) / kinds.sum(1).clip(1)[:, None] - geometry["tcp_world"]
                np.testing.assert_allclose(arrays["hole_anchor"][anchor], expected[anchor], atol=1e-6)
                replay_checks[name] = dict(max_object21_absolute_delta=delta,
                                          parameters=sum(parameter.numel() for parameter in model.parameters()),
                                          checkpoint_dataset_sha256=payload["dataset_sha256"])
            tensors = {key: torch.tensor(arrays[key]) for key in ("object21", "endpoint_geom", "endpoint_residual")}
            inference[condition][name] = dict(arrays=arrays, errors=per_frame_errors(tensors, torch.tensor(geometry["target"])))
            print(f"Frozen checkpoint evaluation complete: {condition}/{name}", flush=True)
    return inference, distributions, replay_checks


def analyze(config):
    """Run verified replay, physical fitting and paired diagnostics in a new output."""
    torch.set_num_threads(config.get("torch_threads", 4))
    geometry = load_geometry(config["validation_geometry"])
    train_geometry = load_geometry(config["train_geometry"])
    identity = json.loads(Path(config["data_identity"]).read_text())
    parent_manifest = json.loads(Path(config["parent_manifest"]).read_text())
    predictions, prediction_hashes = load_aligned_predictions(config["predictions"])
    ids = predictions["baseline"]["sample_ids"]
    lookup = {value: index for index, value in enumerate(geometry["sample_ids"])}
    if len(lookup) != len(ids) or set(lookup) != set(ids):
        raise ValueError("Validation geometry membership must exactly match paired predictions")
    geometry = {key: values[[lookup[value] for value in ids]] for key, values in geometry.items()}
    np.testing.assert_array_equal(geometry["target"], predictions["baseline"]["target"])
    train_episodes = {str(Path(value).parent) for value in train_geometry["sample_ids"]}
    if train_episodes != set(identity["splits"]["train"]):
        raise ValueError("Train geometry must use the complete original train split")
    parents = np.asarray([parent_manifest["parent_by_episode"][str(Path(value).parent)] for value in ids])
    with ExperimentLogger(team="graph", config=config, seed=config.get("seed", 0),
                          model="box_hole_observability", job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "effective_config.json", config)
        states, replay = replay_validation(config, geometry, identity, parent_manifest, output)
        write_json(output / "verified_fixture_states.json", states)
        write_json(output / "simulator_replay.json", replay)
        coverage = observation_geometry(geometry, states)
        groups = diagnostic_groups(predictions["baseline"]["counts"], coverage)
        np.savez_compressed(output / "observation_geometry.npz", sample_ids=ids, **coverage)
        fitting = geometric_oracles(states, config.get("seed", 0))
        with (output / "geometric_fitting.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fitting[0]))
            writer.writeheader()
            writer.writerows(fitting)
        inference, distributions, checks = evaluate_models(config, geometry, states, predictions, output)
        repetitions = config.get("bootstrap_repetitions", 2000)
        summary = {"groups": {}, "oracle_status": {
            "coordinate_replacement": "Unavailable: no certified detector-to-physical identity or original candidate metadata",
            "perfect_geometry_original_visibility": "Unavailable: observed-node membership in physical vertices is unknown",
            "full_gt": "Executed: 8 PEG + 8 BOX exterior + 8 aperture-mouth corners; all target nodes typed BOX",
            "proximity": "3mm candidate support only; never used for replacement or certified physical coverage"}}
        for group, mask in groups.items():
            row = dict(frames=int(mask.sum()), parents=len(set(parents[mask])), conditions={}, paired={})
            for condition, models in inference.items():
                row["conditions"][condition] = {}
                row["paired"][condition] = {}
                for model, result in models.items():
                    row["conditions"][condition][model] = {
                        metric: summarize(result["errors"][metric], parents, mask, repetitions) for metric in METRICS}
                    row["paired"][condition][model] = {
                        metric: summarize(inference["original"][model]["errors"][metric] - result["errors"][metric], parents, mask, repetitions)
                        for metric in METRICS}
            summary["groups"][group] = row
        decomposition = {}
        target = geometry["target"].astype(np.float64)
        for model, result in inference["original"].items():
            values = result["arrays"]
            anchor = values["hole_anchor"].astype(np.float64)
            desired = target[:, 10:13] - anchor
            predicted = values["object21"][:, 10:13].astype(np.float64) - anchor
            signed = predicted - desired
            np.testing.assert_allclose(signed, values["object21"][:, 10:13] - target[:, 10:13], atol=1e-9)
            decomposition[model] = {}
            for group, mask in groups.items():
                metrics = dict(anchor_to_gt_mm=np.linalg.norm(desired, axis=1) * 1000,
                               predicted_offset_mm=np.linalg.norm(predicted, axis=1) * 1000)
                for axis, name in enumerate("xyz"):
                    metrics[f"signed_{name}_mm"] = signed[:, axis] * 1000
                    metrics[f"desired_offset_{name}_mm"] = desired[:, axis] * 1000
                    metrics[f"predicted_offset_{name}_mm"] = predicted[:, axis] * 1000
                local = np.einsum("nij,nj->ni", Rotation.from_quat(target[:, [14, 15, 16, 13]]).as_matrix().transpose(0, 2, 1), signed)
                for axis, name in enumerate("xyz"):
                    metrics[f"signed_hole_local_{name}_mm"] = local[:, axis] * 1000
                decomposition[model][group] = {key: summarize(value, parents, mask, repetitions) for key, value in metrics.items()}
            decomposition[model]["target_present_correlations"] = {
                key: correlation(coverage[key][groups["target_present"]], result["errors"]["hole_position_error_mm"][groups["target_present"]])
                for key in ("half_length_mm", "hole_radius_mm", "yaw_deg", "diameter_mm", "s2_mm", "hole_offset_mm")}
        train_distribution = describe_input(train_geometry["xyz_world"], train_geometry["object_type"], train_geometry["tcp_world"])
        distribution_summary = {}
        for condition, arrays in distributions.items():
            distribution_summary[condition] = {}
            for key, values in arrays.items():
                train_values = train_distribution[key]
                distribution_summary[condition][key] = {**statistics(values), "train": statistics(train_values),
                    "outside_train_range_fraction": float(((values < train_values.min()) | (values > train_values.max())).mean()),
                    "train_min": float(train_values.min()), "train_max": float(train_values.max())}
        summary["input_distributions"] = distribution_summary
        summary["replay_checks"] = checks
        coverage_summary = {group: {key: statistics(values[mask]) for key, values in coverage.items() if values.ndim == 1}
                            for group, mask in groups.items()}
        write_json(output / "summary.json", summary)
        write_json(output / "anchor_offset.json", decomposition)
        write_json(output / "coverage_summary.json", coverage_summary)
        sources = sorted(set([Path(__file__), Path("evaluation/gnn/box_hole_geometry.py"), Path("evaluation/gnn/box_hole_plots.py"),
                              Path("graph/gated_mpnn.py"), Path("evaluation/gnn/vertex_metrics.py"), Path("evaluation/gnn/vertex_diagnostics.py"),
                              Path("training/gnn/trainer.py"), Path("graph/build_graph/graph_batch.py")]))
        snapshot = output / "sources"
        snapshot.mkdir()
        for source in sources:
            destination = snapshot / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        hashes = dict(dataset_sha256=identity["dataset_sha256"], prediction_sha256=prediction_hashes,
                      input_sha256={key: file_sha256(config[key]) for key in ("validation_geometry", "train_geometry", "parent_manifest", "collection_manifest", "data_identity")},
                      checkpoint_sha256={key: file_sha256(path) for key, path in config["checkpoints"].items()},
                      source_sha256={str(path): file_sha256(path) for path in sources})
        write_json(output / "provenance.json", hashes)
        from evaluation.gnn.box_hole_plots import make_plots
        make_plots(output, geometry, states, coverage, groups, inference, fitting)
        logger.record(dataset_sha256=identity["dataset_sha256"], validation_frames=len(ids), validation_parents=len(set(parents)),
                      reserved_test_frames_read=0, backend=replay["backend"], checkpoint_hashes=hashes["checkpoint_sha256"])
        for path in output.rglob("*"):
            if path.is_file():
                logger.link_output(path)
        logger.log({"validation/ab_target_present_hole_mm": summary["groups"]["target_present"]["conditions"]["original"]["combined"]["hole_position_error_mm"]["mean"]})
        print(f"Box/Hole diagnostics saved: {output}", flush=True)
    return str(output)
