"""Measure Object21 changes from adding one or two recent GT corners."""

from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from data_collection.detected_corner_visibility import CONDITIONS, classify_episode, coplanarity_tolerance
from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.five_mode_corner_oracle import VERTICES, read_raw_poses, recent_corner_candidates
from data_collection.storage import write_json
from evaluation.gnn.five_mode_routed import load_model, pinned_path, summarize_mm
from graph.build_graph import NodeType
from training.detected_corner_gnn import graph_batch
from training.gnn.packed_five_mode_data import (
    PackedFrameBatches, indexed_frames, pack_episode, read_archive_member,
)
from training.gnn.tasks import geometry_errors
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


SPLIT = "validation"
METRICS = ("peg_position_error_m", "hole_position_error_m", "endpoint_error_m")
VARIANTS = ("detector_only", "add_one", "add_two")


def validated_episodes(manifest_path, index, raw_catalog, tolerance_m):
    """Join the frozen validation records and check every detector condition."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != "five_mode_supervised_v1":
        raise ValueError("Expected the supervised Five-mode release manifest")
    records = {(row["seed"], row["policy_mode"]): row["split"] for row in manifest["records"]}
    if len(records) != manifest["episodes"]:
        raise ValueError("Duplicate Five-mode episode identities")
    selections = {condition: indexed_frames(index, condition, (SPLIT,), records)
                  for condition in CONDITIONS}
    root = manifest_path.parent
    shard_names = {row["file"] for row in manifest["archives"]}
    raw_index = {row["batch"]: row for row in raw_catalog["archives"]}
    batches = defaultdict(list)
    for row in manifest["records"]:
        if row["split"] == SPLIT:
            batches[row["batch"]].append(row)

    with ExitStack() as stack:
        archives = {}
        for step, (batch, episodes) in enumerate(batches.items(), start=1):
            source = raw_index[batch]
            raw_root = (root / "original_rgbd").resolve()
            raw_path = (root / source["path"]).resolve()
            digest = source["sha256"]
            if (not raw_path.is_relative_to(raw_root) or not raw_path.name.endswith(".tar.zst")
                    or any(episode["raw_sha256"] != digest for episode in episodes)):
                raise ValueError(f"Unexpected raw Five-mode archive: {raw_path}")
            poses = read_raw_poses(raw_path, digest, episodes)
            for episode in episodes:
                key = episode["seed"], episode["policy_mode"]
                tables = {}
                for kind in ("corners", "targets"):
                    receipt = episode["files"][kind]
                    name = receipt["archive"]
                    path = (root / name).resolve()
                    if name not in shard_names or not path.is_relative_to(root) or path.suffix != ".tar":
                        raise ValueError(f"Unexpected Five-mode supervised shard: {path}")
                    if path not in archives:
                        archives[path] = stack.enter_context(tarfile.open(path, "r:"))
                    tables[kind] = read_archive_member(archives[path], receipt, kind)
                groups = classify_episode(tables["corners"], episode["frames"], tolerance_m=tolerance_m)
                if any(frames != selections[name][key] for name, frames in groups.items()):
                    raise ValueError(f"Current detector classification differs from the release: {key}")
                yield episode, tables, poses[key], groups
            if step % 5 == 0 or step == len(batches):
                print(f"Verified detector conditions and poses in {step}/{len(batches)} raw batches", flush=True)


def load_validation(config, manifest_path, index, raw_catalog, device):
    """Pack observed graphs and save the ranked candidates separately from detector nodes."""
    tolerance = coplanarity_tolerance(config["coplanarity_tolerance_m"])
    packed, conditions, counts, vertex_ids, ages, points = [], [], [], [], [], []
    for episode, tables, poses, groups in validated_episodes(manifest_path, index, raw_catalog, tolerance):
        frames = np.arange(episode["frames"])
        arrays = pack_episode(episode, tables, include_surface=False)
        xyz, kinds, tcp, target = arrays
        np.testing.assert_allclose(poses[:, :3] - tcp, target[:, :3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(poses[:, 14:17] - tcp, target[:, 10:13], atol=1e-6, rtol=0)
        packed.append((episode, arrays, frames))
        labels = np.empty(len(frames), dtype="U11")
        for condition, rows in groups.items():
            labels[rows] = condition
        conditions.extend(labels)
        for ids, elapsed, gt_xyz in recent_corner_candidates(tables["corners"], poses, **config["candidates"]):
            count = min(2, len(ids))
            counts.append(len(ids))
            vertex_ids.append(np.pad(ids[:count], (0, 2 - count), constant_values=-1))
            ages.append(np.pad(elapsed[:count], (0, 2 - count)))
            points.append(np.pad(gt_xyz[:count], ((0, 2 - count), (0, 0))))
    loader = PackedFrameBatches(packed, device, config["batch_size"], None)
    conditions = np.asarray(conditions)
    actual = {name: int((conditions == name).sum()) for name in CONDITIONS}
    if len(set(loader.ids)) != len(loader.ids) or actual != config["expected_frames"]:
        raise ValueError(f"Unexpected validation frame coverage: {actual}")
    return loader, conditions, np.asarray(counts), np.stack(vertex_ids), np.stack(ages), np.stack(points)


@torch.inference_mode()
def predict(model, loader, ids, points):
    """Run the frozen checkpoint, adding candidate coordinates in the original graph format."""
    device = loader.device
    kinds = torch.as_tensor([vertex[0] for vertex in VERTICES], device=device)
    ids = torch.as_tensor(ids, device=device)
    candidate_kinds = torch.where(ids >= 0, kinds[ids.clamp_min(0)], NodeType.PAD)
    points = torch.as_tensor(points, dtype=loader.xyz.dtype, device=device)

    def forward(xyz, node_kinds, tcp):
        valid = node_kinds != NodeType.PAD
        graph, _ = graph_batch(xyz, node_kinds, valid, valid, tcp)
        return model(graph)["object21"]

    batches = {variant: [] for variant in VARIANTS}
    for rows in loader.frame_order().split(loader.batch_size):
        rows = rows.to(device)
        xyz, types, tcp = loader.xyz[rows], loader.kinds[rows].long(), loader.tcp[rows]
        baseline = forward(xyz, types, tcp)
        batches[VARIANTS[0]].append(baseline.cpu())
        for budget, variant in enumerate(VARIANTS[1:], start=1):
            present = ids[rows, 0] >= 0
            prediction = baseline.clone()
            if present.any():
                add_points = points[rows[present], :budget]
                add_kinds = candidate_kinds[rows[present], :budget]
                prediction[present] = forward(torch.cat((xyz[present], add_points), dim=1),
                                              torch.cat((types[present], add_kinds), dim=1), tcp[present])
            batches[variant].append(prediction.cpu())
    return {name: torch.cat(values) for name, values in batches.items()}


def summarize(conditions, counts, errors):
    """Report all frames as well as paired subsets with enough candidates."""
    results = {}
    for condition in CONDITIONS:
        mask = conditions == condition
        row = {"frames": int(mask.sum()), "with_one_or_more": int((mask & (counts >= 1)).sum()),
               "with_two_or_more": int((mask & (counts >= 2)).sum())}
        for cohort, eligible in (("all", mask), ("exact_one", mask & (counts >= 1)),
                                 ("exact_two", mask & (counts >= 2))):
            variants = ("detector_only", "add_one") if cohort == "exact_one" else VARIANTS
            row[cohort] = {
                name: {metric: summarize_mm(errors[name][eligible, index])
                       for index, metric in enumerate(METRICS)} for name in variants
            } if eligible.any() else {}
        results[condition] = row
    return results


def evaluate(config):
    """Evaluate the pinned detector-only Condition 1 checkpoint on validation."""
    manifest = pinned_path(config["manifest"])
    index = json.loads(pinned_path(config["condition_index"]).read_text())
    raw_catalog = json.loads(pinned_path(config["raw_catalog"]).read_text())
    device = set_device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    checkpoint = pinned_path(config["checkpoint"])
    checkpoint_config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]["data"]
    if checkpoint_config["include_surface"] or checkpoint_config["condition"] != "condition_1":
        raise ValueError("Expected the frozen detector-only Condition 1 checkpoint")
    model = load_model(config["checkpoint"], device).requires_grad_(False)
    with ExperimentLogger(team="graph", model="vertex_edge_gated_mpnn", config=config,
                          job_type="evaluate") as logger:
        logger.link_output(checkpoint, checkpoint=True)
        logger.record(dataset_sha256=config["manifest"]["sha256"], device=str(device),
                      torch_version=str(torch.__version__), evaluator_sha256=file_sha256(Path(__file__)))
        loader, conditions, counts, ids, ages, points = load_validation(config, manifest, index, raw_catalog, device)
        print(f"Running paired inference on {len(loader.ids):,} validation frames", flush=True)
        predictions = predict(model, loader, ids, points)
        target = loader.target.cpu()
        errors = {}
        for name, prediction in predictions.items():
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Nonfinite Object21 prediction: {name}")
            metrics = geometry_errors(prediction, target)
            errors[name] = np.column_stack([metrics[metric].numpy() * 1000 for metric in METRICS])
        results = summarize(conditions, counts, errors)

        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        trace = output / "validation_errors.npz"
        np.savez_compressed(trace, frame_ids=np.asarray(loader.ids), conditions=conditions,
                            candidates=counts, candidate_vertex_ids=ids, candidate_ages=ages,
                            candidate_xyz=points, metric_order=np.asarray(METRICS), **errors)
        logger.link_output(trace)
        path = output / "results.json"
        write_json(path, dict(run_id=logger.run_id, split=SPLIT, config=config,
                              frame_ids_sha256=canonical_hash(loader.ids), candidate_vertex_order=VERTICES,
                              results=results))
        logger.link_output(path, kind="metrics")
        logger.log(results, step=1, prefix="validation")
        print(f"Completed {path}", flush=True)
