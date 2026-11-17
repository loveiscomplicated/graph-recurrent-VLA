"""Differentiable episode replay with GT-free decisions and separate identity audit."""

import numpy as np
import torch

from data_collection.temporal.corner_tracks import masked_frame
from graph.temporal.corner_tracking.tracker import CornerTracker
from graph.temporal.corner_tracking.graph import batch_completed_graphs
from training.gnn.loss import object21_loss


def replay_episode(motion, gnn, episode, plan, calibration, settings, loss_settings=None,
                   *, collect_diagnostics=True, compute_object_loss=True):
    """Keep complete-episode gradients; assign GT identities only in this audit.

    The tracker never receives frame.physical_ids/reference_xyz. A changed
    certified identity is recorded as an error; it does not repair the tracker.
    """
    tracker = CornerTracker(motion, calibration, settings)
    bindings, last_certified, pending_recovery = {}, {}, {}
    last_track_for_reference, seen_references = {}, set()
    node_losses, object_losses, position_errors = [], [], []
    outputs, targets, endpoint_geometry, endpoint_residuals, node_rows, trace = [], [], [], [], [], []
    graph_batch_size = getattr(gnn, "corner_graph_batch_size", 1)
    graph_buffer, target_buffer = [], []

    def evaluate_graph_buffer():
        """Evaluate independent frames together without changing causal tracking."""
        graph = batch_completed_graphs(graph_buffer) if len(graph_buffer) > 1 else graph_buffer[0]
        # A final single-frame chunk still needs validation after deferred checks.
        if len(graph_buffer) == 1 and graph_batch_size > 1:
            graph.geometry.validate()
        output = gnn(graph.geometry, graph.features())
        target = motion.tensor(np.stack(target_buffer))
        loss = object21_loss(output, target, **(loss_settings or {}))["total"]
        object_losses.append((loss, len(graph_buffer)))
        if collect_diagnostics:
            outputs.append(output["object21"].detach().cpu())
            targets.append(target.detach().cpu())
            endpoint_geometry.append(output["endpoint_geom"].detach().cpu())
            endpoint_residuals.append(output["endpoint_residual"].detach().cpu())
        graph_buffer.clear()
        target_buffer.clear()
    counters = dict(id_switches=0, track_identity_changes=0, duplicate_tracks=0,
                    reobservations=0, false_reconnections=0, unrecovered_reappearances=0,
                    adopted_predictions=0, eligible_predictions=0, unknown_prior_targets=0,
                    ambiguous_detections=0, retired_tracks=0, certified_assignments=0,
                    certified_missing_corner_frames=0, covered_missing_corner_frames=0,
                    node_prior_count=0, rejected_identity_supervision=0)
    recovery_frames = []
    previous_missing = set()
    for time, frame in enumerate(episode.frames):
        observation, identities, original_indices = masked_frame(frame, plan, time)
        result = tracker.step(observation, include_graph=collect_diagnostics or compute_object_loss,
                              validate_graph=graph_batch_size == 1)
        accepted = dict(result.assignments)
        accepted.update(result.births)
        for identifier, index in accepted.items():
            certified = int(identities[index])
            if certified < 0:
                continue
            counters["certified_assignments"] += 1
            previous = last_certified.get(identifier)
            if previous is not None and previous != certified:
                counters["track_identity_changes"] += 1
            last_certified[identifier] = certified
            bindings.setdefault(identifier, certified)
            if identifier in previous_missing:
                counters["reobservations"] += 1
                if bindings[identifier] != certified:
                    counters["false_reconnections"] += 1
                else:
                    recovery_frames.append(time - pending_recovery.pop(identifier, time))
            if tracker.tracks[identifier].confirmed:
                previous_track = last_track_for_reference.get(certified)
                if previous_track is not None and previous_track != identifier:
                    counters["id_switches"] += 1
                last_track_for_reference[certified] = identifier
        active_identities = [bindings[key] for key, track in tracker.tracks.items()
                             if track.confirmed and key in bindings]
        counters["duplicate_tracks"] += len(active_identities) - len(set(active_identities))
        currently_certified = set(map(int, identities[identities >= 0]))
        missing_references = seen_references - currently_certified
        counters["certified_missing_corner_frames"] += len(missing_references)
        covered = {bindings[identifier] for identifier in result.adopted_predictions
                   if identifier in bindings and last_certified.get(identifier) == bindings[identifier]}
        counters["covered_missing_corner_frames"] += len(covered & missing_references)
        seen_references.update(currently_certified)
        supervised_priors, supervised_references = [], []
        diagnostic_priors, diagnostic_references, diagnostic_metadata = [], [], []
        for identifier, prior in result.priors.items():
            counters["node_prior_count"] += 1
            reference = bindings.get(identifier)
            accepted_index = result.assignments.get(identifier)
            disagreement = (accepted_index is not None and identities[accepted_index] >= 0
                            and int(identities[accepted_index]) != reference)
            if reference is None:
                counters["unknown_prior_targets"] += 1
                continue
            # Certifiable ID changes stay in metrics, but do not become false
            # coordinate supervision for a mismatched observation anchor.
            if not disagreement and last_certified.get(identifier) == reference:
                supervised_priors.append(prior)
                supervised_references.append(reference)
            else:
                counters["rejected_identity_supervision"] += 1
            if collect_diagnostics:
                diagnostic_priors.append(prior)
                diagnostic_references.append(reference)
                diagnostic_metadata.append(dict(frame=time, track_id=identifier, reference=reference,
                                                kind=prior.kind, age=prior.age,
                                                adopted=identifier in result.adopted_predictions,
                                                identity_consistent=not disagreement and last_certified.get(identifier) == reference))
        if supervised_priors:
            positions = torch.stack([prior.position for prior in supervised_priors])
            variance = torch.stack([prior.prediction.variance for prior in supervised_priors])
            error = positions - motion.tensor(frame.reference_xyz[supervised_references])
            position_errors.append(error.detach().norm(dim=-1))
            position_unit = (loss_settings or {}).get("pos_unit_m", 0.01)
            node_losses.append(0.5 * ((variance / position_unit ** 2).log()
                                      + error.square() / variance).mean(dim=-1))
        if diagnostic_priors:
            positions = torch.stack([prior.position for prior in diagnostic_priors])
            variance = torch.stack([prior.prediction.variance for prior in diagnostic_priors])
            errors = (positions - motion.tensor(frame.reference_xyz[diagnostic_references])).norm(dim=-1)
            statistics = torch.stack((errors, variance.sum(-1).sqrt()), dim=-1).detach().cpu().tolist()
            for metadata, (error, deviation) in zip(diagnostic_metadata, statistics):
                node_rows.append(dict(**metadata, error_m=error, std_norm_m=deviation))
        if compute_object_loss or collect_diagnostics:
            graph_buffer.append(result.graph)
            target_buffer.append(frame.target_object21)
            if len(graph_buffer) == graph_batch_size:
                evaluate_graph_buffer()
        missing = {identifier for identifier, status in result.statuses.items()
                   if status in ("missing", "dormant")}
        # Recovery delay starts when the known physical point actually returns
        # to detector output, not at synthetic-mask end or GT visibility change.
        for identifier in missing:
            reference = bindings.get(identifier)
            if reference is not None and reference in identities:
                pending_recovery.setdefault(identifier, time)
        previous_missing = missing
        counters["adopted_predictions"] += len(result.adopted_predictions)
        counters["eligible_predictions"] += len(result.eligible_predictions)
        counters["ambiguous_detections"] += len(result.ambiguous_detections)
        counters["retired_tracks"] += len(result.retired)
        if collect_diagnostics:
            trace.append(dict(sample_id=frame.sample_id, statuses=result.statuses,
                              assignments={key: int(original_indices[value]) for key, value in result.assignments.items()},
                              births={key: int(original_indices[value]) for key, value in result.births.items()},
                              adopted=result.adopted_predictions,
                              ambiguous_detections=sorted(result.ambiguous_detections), retired=result.retired))
        for identifier in result.retired:
            if identifier in pending_recovery:
                counters["unrecovered_reappearances"] += 1
                pending_recovery.pop(identifier)
    if graph_buffer:
        evaluate_graph_buffer()
    counters["unrecovered_reappearances"] += len(pending_recovery)
    zero_parameter = next(motion.parameters(), None) if isinstance(motion, torch.nn.Module) else None
    if zero_parameter is None:
        zero_parameter = next(gnn.parameters())
    node_loss = (torch.cat(node_losses).mean() if node_losses
                 else zero_parameter.sum() * 0)
    object_loss = (torch.stack([loss * count for loss, count in object_losses]).sum()
                   / sum(count for _, count in object_losses) if object_losses else None)
    position_error_m = torch.cat(position_errors).mean() if position_errors else None
    return dict(node_loss=node_loss, object_loss=object_loss, position_error_m=position_error_m,
                prediction=torch.cat(outputs).numpy() if outputs else None,
                target=torch.cat(targets).numpy() if targets else None,
                endpoint_geom=torch.cat(endpoint_geometry).numpy() if endpoint_geometry else None,
                endpoint_residual=torch.cat(endpoint_residuals).numpy() if endpoint_residuals else None,
                nodes=node_rows, counters=counters, trace=trace, recovery_frames=recovery_frames,
                node_supervision_count=sum(loss.numel() for loss in node_losses), frames=len(episode.frames))
