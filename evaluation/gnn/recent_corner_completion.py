"""Compare hard learned selection and recency rules on matched Five-mode frames."""

from pathlib import Path

import numpy as np
import torch

from data_collection.detected_corner_visibility import CONDITIONS
from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.five_mode_recent_corners import load_cohort
from data_collection.storage import write_json
from evaluation.gnn.five_mode_corner_oracle import METRICS
from evaluation.gnn.five_mode_routed import load_model, pinned_path, summarize_mm
from graph.temporal.recent_corner_completion import RecentCornerCompletion, run_actions
from training.gnn.loss import object21_loss
from training.gnn.tasks import geometry_errors
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


VARIANTS = ("detector_only", "learned", "recent_one", "recent_two", "recent_matched_count")


def per_frame_loss(output, target):
    """Evaluate the existing Object21 objective independently for every frame."""
    def one(prediction, residual, label):
        return object21_loss({"object21": prediction.unsqueeze(0),
                              "endpoint_residual": residual.unsqueeze(0)}, label.unsqueeze(0))["total"]

    return torch.vmap(one)(output["object21"], output["endpoint_residual"], target)


def select_rows(dataset, indices, device):
    """Transfer and trim a batch, keeping at least one padded observed slot."""
    batch = {name: array[indices].to(device) for name, array in dataset.items()}
    width = max(1, int((batch["node_kinds"] >= 0).sum(dim=1).max()))
    for key in ("xyz", "node_kinds", "observed_families"):
        batch[key] = batch[key][:, :width]
    return batch


@torch.no_grad()
def compare(gnn, completion, dataset, indices, device, batch_size, *, learned_only=False,
            completion_conditions=None):
    """Return aligned mean-ready error arrays and the number of selected nodes."""
    completion.eval()
    sizes = (completion.actions >= 0).sum(dim=-1)
    first_pair = int((sizes == 2).nonzero()[0])
    recency_actions = sizes.new_tensor([0, 1, first_pair])
    variants = ("learned",) if learned_only else VARIANTS
    traces = {variant: [] for variant in variants}
    labels = None if completion_conditions is None else sizes.new_tensor(completion_conditions)
    for ids in indices.split(batch_size):
        batch = select_rows(dataset, ids, device)
        available = (batch["candidate_kinds"] >= 0).sum(dim=1)
        if labels is None:
            recovered, logits, _ = completion(batch)
            selected = logits.argmax(dim=1)
        else:
            active = torch.isin(batch["condition"], labels)
            recovered = batch["anchors"].clone()
            selected = torch.zeros_like(available)
            available = available.masked_fill(~active, 0)
            if active.any():
                active_batch = {name: array[active] for name, array in batch.items()}
                positions, logits, _ = completion(active_batch)
                recovered[active] = positions
                selected[active] = logits.argmax(dim=1)
        if learned_only:
            decisions = selected[None]
        else:
            decisions = torch.stack((
                torch.zeros_like(selected), selected, recency_actions[available.clamp_max(1)],
                recency_actions[available.clamp_max(2)], recency_actions[sizes[selected]],
            ))
        needed = torch.zeros((len(ids), len(sizes)), dtype=torch.bool, device=device)
        needed.scatter_(1, decisions.T, True)
        rows, choices = needed.nonzero(as_tuple=True)
        output = run_actions(gnn, batch, recovered, completion.actions[choices], rows)
        target = batch["target"][rows]
        losses = per_frame_loss(output, target)
        metrics = geometry_errors(output["object21"], target)
        values = torch.column_stack((losses, *(1000 * metrics[name] for name in METRICS)))
        gathered = values.new_zeros((*needed.shape, values.shape[-1]))
        gathered[rows, choices] = values
        all_rows = torch.arange(len(ids), device=device)
        for variant, choice in zip(variants, decisions, strict=True):
            result = torch.cat((gathered[all_rows, choice], sizes[choice, None]), dim=1)
            traces[variant].append(result.cpu().numpy())
    traces = {variant: np.concatenate(parts) for variant, parts in traces.items()}
    if not all(np.isfinite(values).all() for values in traces.values()):
        raise ValueError("Nonfinite Object21 evaluation metrics")
    return traces


def summarize(traces, conditions, available):
    """Report physical errors, loss and selection count for each condition."""
    results = {}
    for label, name in enumerate(CONDITIONS, start=1):
        mask = conditions == label
        row = dict(frames=int(mask.sum()), with_candidates=int((mask & (available > 0)).sum()))
        for scope, selected in (("all", mask), ("candidate_frames", mask & (available > 0))):
            if not selected.any():
                continue
            row[scope] = {}
            for variant, trace in traces.items():
                values = trace[selected]
                row[scope][variant] = {
                    "object21_loss": float(values[:, 0].mean(dtype=np.float64)),
                    **{metric: summarize_mm(values[:, idx]) for idx, metric in enumerate(METRICS, start=1)},
                    "selected_counts": np.bincount(values[:, -1].astype(int), minlength=3).tolist(),
                }
        results[name] = row
    return results


def evaluate_checkpoint(config, gnn, completion, dataset, frame_ids, device, checkpoint, *,
                        split="validation", completion_conditions=None):
    """Evaluate the selected checkpoint and save trace and summary with their own run ID."""
    with ExperimentLogger(team="graph", model="recent_corner_completion", job_type="evaluate",
                          config={**config, "completion_checkpoint": str(checkpoint)}) as logger:
        logger.link_output(config["frozen_gnn"]["path"], checkpoint=True)
        logger.record(device=str(device), frozen_gnn_run_id=logger.metadata.get("source_run_id"),
                      dataset_sha256=config["data"]["manifest"]["sha256"],
                      checkpoint_sha256=file_sha256(checkpoint), split=split)
        logger.link_output(checkpoint, checkpoint=True)
        traces = compare(gnn, completion, dataset, torch.arange(len(frame_ids)), device,
                         config["evaluation"]["batch_size"], completion_conditions=completion_conditions)
        conditions = dataset["condition"].numpy()
        available = (dataset["candidate_kinds"].numpy() >= 0).sum(axis=1)
        if completion_conditions is not None:
            available = np.where(np.isin(conditions, completion_conditions), available, 0)
        results = summarize(traces, conditions, available)
        root = Path(config["output_root"]) / logger.run_id
        root.mkdir(parents=True, exist_ok=False)
        trace_path = root / f"{split}_errors.npz"
        np.savez_compressed(trace_path, frame_ids=np.asarray(frame_ids), condition=conditions,
                            candidate_count=available, columns=np.asarray(("object21_loss", *METRICS, "selected_count")),
                            **traces)
        logger.link_output(trace_path)
        path = root / "results.json"
        write_json(path, dict(run_id=logger.run_id, checkpoint=str(checkpoint), device=str(device), split=split,
                              completion_conditions=completion_conditions,
                              frame_ids_sha256=canonical_hash(frame_ids), results=results))
        logger.link_output(path, kind="metrics")
        logger.log(results, step=1, prefix=split)
        print(f"Completed {split} comparison: {path}", flush=True)


def load_test_inputs(config):
    """Load the pinned completion model and reproduce its test cohort."""
    checkpoint = pinned_path(config["checkpoint"])
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if snapshot["epoch"] != config["expected_epoch"]:
        raise ValueError("Unexpected corner-completion checkpoint epoch")
    train_config = snapshot["config"]
    effective = {**train_config, **config, "data": {**train_config["data"], **config["data"]}}
    device = set_device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    completion = RecentCornerCompletion(effective["association"]["max_candidates"], **effective["model"])
    completion.load_state_dict(snapshot["model"], strict=True)
    completion.to(device).eval().requires_grad_(False)
    dataset, frame_ids = load_cohort(effective, "test")
    counts = torch.bincount(dataset["condition"], minlength=4)[1:].tolist()
    if dict(zip(CONDITIONS, counts, strict=True)) != config["expected_frames"]:
        raise ValueError(f"Unexpected test frame counts by condition: {counts}")
    return effective, device, checkpoint, completion, dataset, frame_ids


def evaluate_test(config):
    """Replay the saved model and detector-only GNN with test-only routing."""
    effective, device, checkpoint, completion, dataset, frame_ids = load_test_inputs(config)
    gnn = load_model(effective["frozen_gnn"], device).requires_grad_(False)
    evaluate_checkpoint(effective, gnn, completion, dataset, frame_ids, device, checkpoint,
                        split="test", completion_conditions=config["completion_conditions"])
