"""Identify which Hole rim candidates reach Object21 on Condition 3 test frames."""

from pathlib import Path

import numpy as np
import torch

from data_collection.storage import write_json
from evaluation.gnn.five_mode_routed import load_model, pinned_path, summarize_mm
from evaluation.gnn.recent_corner_completion import VARIANTS, load_test_inputs, select_rows
from graph.temporal.recent_corner_completion import run_actions
from main import load_config
from utils.experiment_logger import ExperimentLogger


@torch.no_grad()
def replay_selection(gnn, completion, dataset, device, batch_size, active_conditions):
    """Recover the selected detector families and verify their Hole predictions."""
    trace = {name: [] for name in ("selected_families", "predicted_hole_error_mm", "zero_hole_anchor")}
    labels = torch.tensor(active_conditions, device=device)
    for rows in torch.arange(len(dataset["condition"])).split(batch_size):
        batch = select_rows(dataset, rows, device)
        active = torch.isin(batch["condition"], labels)
        current = batch["condition"][active] == 3
        if not current.any():
            continue
        batch = {name: values[active] for name, values in batch.items()}
        recovered, logits, _ = completion(batch)
        indices = current.nonzero(as_tuple=True)[0]
        options = completion.actions[logits[indices].argmax(dim=1)]
        families = batch["candidate_kinds"][indices[:, None], options.clamp_min(0)]
        families = families.masked_fill(options < 0, -1)
        output = run_actions(gnn, batch, recovered, options, indices)
        delta = output["object21"][:, 10:13] - batch["target"][indices, 10:13]
        trace["selected_families"].append(families.cpu().numpy())
        trace["predicted_hole_error_mm"].append((delta.norm(dim=1) * 1000).cpu().numpy())
        trace["zero_hole_anchor"].append((output["hole_anchor"] == 0).all(dim=1).cpu().numpy())
    return {name: np.concatenate(chunks) for name, chunks in trace.items()}


def prior_rim_age(frame_ids, observed_rim_count, max_age):
    """Find the most recent real rim detection strictly before the current frame."""
    ages = np.full(len(frame_ids), -1, dtype=np.int8)
    last_rim = {}
    for row, frame_id in enumerate(frame_ids):
        episode, _, frame_number = frame_id.rpartition(":")
        frame = int(frame_number)
        previous = last_rim.get(episode)
        if previous is not None and 1 <= frame - previous <= max_age:
            ages[row] = frame - previous
        if observed_rim_count[row] > 0:
            last_rim[episode] = frame
    return ages


def summarize_groups(groups, errors):
    """Report paired errors and the group's contribution to the learned mean."""
    total = errors["learned"].sum(dtype=np.float64)
    return {name: {
        "frames": int(mask.sum()),
        "learned_error_share": float(errors["learned"][mask].sum(dtype=np.float64) / total),
        "hole_center_error": {variant: summarize_mm(values[mask]) for variant, values in errors.items()}
        if mask.any() else {},
    } for name, mask in groups.items()}


def diagnose(config):
    """Join recoverable rim points and the saved paired test errors by frame ID."""
    test = load_config(config["test_config"])
    effective, device, checkpoint, completion, dataset, frame_ids = load_test_inputs(test)
    gnn = load_model(effective["frozen_gnn"], device).requires_grad_(False)
    recorded_path = pinned_path(config["reference_trace"])
    conditions = dataset["condition"].numpy()
    condition_3 = conditions == 3
    with np.load(recorded_path, allow_pickle=False) as reference:
        np.testing.assert_array_equal(reference["frame_ids"], frame_ids)
        np.testing.assert_array_equal(reference["condition"], conditions)
        column = reference["columns"].tolist().index("hole_position_error_m")
        errors = {variant: reference[variant][condition_3, column] for variant in VARIANTS}
        recorded_selection = reference["learned"][condition_3, -1]
        recorded_candidates = reference["candidate_count"][condition_3]

    replay = replay_selection(gnn, completion, dataset, device, test["evaluation"]["batch_size"],
                              test["completion_conditions"])
    selected = replay["selected_families"]
    candidates = dataset["candidate_kinds"].numpy()[condition_3]
    np.testing.assert_array_equal((candidates >= 0).sum(axis=1), recorded_candidates)
    np.testing.assert_array_equal((selected >= 0).sum(axis=1), recorded_selection)
    np.testing.assert_allclose(replay["predicted_hole_error_mm"], errors["learned"], rtol=0, atol=0.01)

    observed_families = dataset["observed_families"].numpy()
    rim_count = (observed_families == 2).sum(axis=1)
    rim_age = prior_rim_age(frame_ids, rim_count, effective["association"]["max_age"])[condition_3]
    current_rim = rim_count[condition_3] > 0
    box_count = (observed_families[condition_3] == 1).sum(axis=1)
    rim_candidate = (candidates == 2).any(axis=1)
    selected_rim = (selected == 2).any(axis=1)
    current_target = current_rim | (box_count > 0)
    selected_target = (selected > 0).any(axis=1)
    no_target = ~current_target & ~selected_target
    np.testing.assert_array_equal(no_target, replay["zero_hole_anchor"])
    if (selected_rim & ~rim_candidate).any() or (rim_candidate & (rim_age < 0)).any():
        raise ValueError("A rim selection or candidate is missing its detector history")

    missed_shortlist = ~current_rim & (rim_age > 0) & ~rim_candidate
    if not (recorded_candidates[missed_shortlist] == effective["association"]["max_candidates"]).all():
        raise ValueError("A recent rim detection with no current matches did not enter a nonfull shortlist")
    selection_groups = {
        "no_rim_candidate": ~rim_candidate,
        "rim_candidate_skipped": rim_candidate & ~selected_rim,
        "rim_candidate_selected": selected_rim,
    }
    support_groups = {
        "no_observed_or_selected_target": no_target,
        "target_from_completion_only": ~current_target & selected_target,
        "current_target_observation": current_target,
    }
    visibility_groups = {
        "current_rim": current_rim,
        "no_current_or_recent_rim": ~current_rim & (rim_age < 0),
        "no_current_rim_excluded_from_shortlist": missed_shortlist,
        "no_current_rim_candidate_skipped": ~current_rim & rim_candidate & ~selected_rim,
        "no_current_rim_candidate_selected": ~current_rim & selected_rim,
    }
    results = {
        "selection": summarize_groups(selection_groups, errors),
        "hole_graph_support": summarize_groups(support_groups, errors),
        "detector_rim_visibility": summarize_groups(visibility_groups, errors),
    }
    with ExperimentLogger(team="graph", model="recent_corner_completion", job_type="evaluate",
                          config={**test, **config}) as logger:
        logger.link_output(checkpoint, checkpoint=True)
        logger.record(device=str(device), split="test", dataset_sha256=effective["data"]["manifest"]["sha256"],
                      reference_trace_sha256=config["reference_trace"]["sha256"])
        root = Path(test["output_root"]) / logger.run_id
        root.mkdir(parents=True, exist_ok=False)
        trace_path = root / "condition_3_hole_support.npz"
        np.savez_compressed(trace_path, frame_ids=np.asarray(frame_ids)[condition_3], candidates=candidates,
                            selected=selected, rim_count=rim_count[condition_3], box_count=box_count,
                            prior_rim_age=rim_age, hole_error_replay_mm=replay["predicted_hole_error_mm"],
                            **errors)
        logger.link_output(trace_path)
        path = root / "results.json"
        write_json(path, dict(run_id=logger.run_id, frames=int(condition_3.sum()), results=results))
        logger.link_output(path, kind="metrics")
        logger.log(results, step=1, prefix="condition_3")
        print(f"Completed Condition 3 Hole diagnostic: {path}", flush=True)
