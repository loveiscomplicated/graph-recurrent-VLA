"""Paired complete-episode reports for the corner-before-GNN experiment."""

import numpy as np
import torch

from evaluation.gnn.vertex_metrics import per_frame_errors


def distribution(values):
    """Keep absent populations explicit rather than reporting zero error."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, mean=None, median=None, p90=None, p95=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, 0.90)), p95=float(np.quantile(values, 0.95)))


def summarize(results, calibration):
    """Report physical errors, identity failures and adoption denominators together."""
    predictions = torch.from_numpy(np.concatenate([result["prediction"] for result in results]))
    targets = torch.from_numpy(np.concatenate([result["target"] for result in results]))
    output = dict(object21=predictions,
                  endpoint_geom=torch.from_numpy(np.concatenate([result["endpoint_geom"] for result in results])),
                  endpoint_residual=torch.from_numpy(np.concatenate([result["endpoint_residual"] for result in results])))
    physical = {key: distribution(value) for key, value in per_frame_errors(output, targets).items()}
    counters = {key: sum(result["counters"][key] for result in results)
                for key in results[0]["counters"]}
    rows = [row for result in results for row in result["nodes"]]
    by_age = {}
    for low, high in ((1, 2), (3, 8), (9, 16), (17, 32)):
        selected = [row for row in rows if low <= row["age"] <= high]
        by_age[f"{low}_{high}"] = dict(
            error_m=distribution([row["error_m"] for row in selected]),
            adopted=sum(row["adopted"] for row in selected),
            inconsistent_identity=sum(not row["identity_consistent"] for row in selected))
    covered = []
    for row in rows:
        factor = calibration.scale(row["kind"], row["age"])
        if np.isfinite(factor) and row["identity_consistent"]:
            covered.append(row["error_m"] <= factor * row["std_norm_m"])
    eligible = counters["eligible_predictions"]
    reobservations = counters["reobservations"]
    missing_references = counters["certified_missing_corner_frames"]
    supervised = sum(result["node_supervision_count"] for result in results)
    return dict(
        frames=sum(result["frames"] for result in results), episodes=len(results),
        object21=physical, node_position_error_m=distribution([row["error_m"] for row in rows]),
        node_error_by_age=by_age, tracking=counters,
        node_supervision_count=supervised,
        node_supervision_fraction=(supervised / counters["node_prior_count"]
                                   if counters["node_prior_count"] else None),
        adoption_fraction=counters["adopted_predictions"] / eligible if eligible else None,
        certified_missing_coverage=(counters["covered_missing_corner_frames"] / missing_references
                                    if missing_references else None),
        false_reconnection_fraction=(counters["false_reconnections"] / reobservations
                                     if reobservations else None),
        recovery_delay_frames=distribution([value for result in results for value in result["recovery_frames"]]),
        uncertainty95_coverage=float(np.mean(covered)) if covered else None,
        uncertainty95_count=len(covered),
        object21_loss=float(np.mean([float(result["object_loss"].detach().cpu()) for result in results])),
        node_nll=float(np.mean([float(result["node_loss"].detach().cpu()) for result in results])),
    )


def paired_comparison(runs, episodes, bootstrap_samples=1000):
    """Average seed errors per frame, then bootstrap paired parent clusters."""
    parents = np.asarray([episode.parent for episode in episodes for _ in episode.frames])
    unique_parents = sorted(set(parents))
    result = {}
    for scenario in ("natural", "masked", "end_stress"):
        methods = {}
        reference_ids = reference_target = None
        for run in runs:
            with np.load(f"{run['output']}/{scenario}/predictions.npz", allow_pickle=False) as data:
                if reference_ids is None:
                    reference_ids, reference_target = data["sample_ids"].copy(), data["target"].copy()
                np.testing.assert_array_equal(reference_ids, data["sample_ids"])
                np.testing.assert_array_equal(reference_target, data["target"])
                output = {key: torch.from_numpy(data[name].copy()) for key, name in
                          (("object21", "prediction"), ("endpoint_geom", "endpoint_geom"),
                           ("endpoint_residual", "endpoint_residual"))}
                errors = per_frame_errors(output, torch.from_numpy(data["target"].copy()))
                methods.setdefault(run["architecture"], []).append(errors)
        means = {method: {key: np.mean([seed[key] for seed in seeds], axis=0)
                          for key in seeds[0]} for method, seeds in methods.items()}
        summary = dict(methods={method: {key: distribution(value) for key, value in errors.items()}
                                for method, errors in means.items()},
                       parents=len(unique_parents), seed_counts={key: len(value) for key, value in methods.items()})
        if set(means) == {"mamba", "transformer"}:
            differences = {key: means["mamba"][key] - means["transformer"][key] for key in means["mamba"]}
            rng = np.random.default_rng(0)
            parent_indices = [np.flatnonzero(parents == parent) for parent in unique_parents]
            intervals = {}
            for key, values in differences.items():
                draws = []
                if len(unique_parents) >= 2:
                    for _ in range(bootstrap_samples):
                        sample = rng.integers(len(parent_indices), size=len(parent_indices))
                        indices = np.concatenate([parent_indices[index] for index in sample])
                        draws.append(float(values[indices].mean()))
                intervals[key] = dict(mean=float(values.mean()),
                                      parent_bootstrap95=(np.quantile(draws, [0.025, 0.975]).tolist()
                                                          if draws else None))
            summary["mamba_minus_transformer"] = intervals
        result[scenario] = summary
    return dict(scenarios=result,
                interpretation="Mean of seed errors per frame; paired parent bootstrap; development validation, not independent test")


def evaluate(config):
    """Root-main evaluation entrypoint; load both networks and saved calibration."""
    from training.gnn.temporal.corner_tracking import evaluate_checkpoint
    return evaluate_checkpoint(config)
