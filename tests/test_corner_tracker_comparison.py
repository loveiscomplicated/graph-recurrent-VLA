"""Common-cohort scoring, causal identity binding and input-isolation contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.corner_tracker_comparison import (
    PhysicalTrackCollector,
    audit_episode_inputs,
    causal_masks,
    checked_file,
    comparison_metrics,
    error_summary,
    physical_observations,
)
from tests.test_spatial_temporal_corner_tracker import prepared_episode_arrays


def test_common_mask_excludes_first_sighting_and_resets_per_episode():
    observed = np.array([[False, False], [True, False], [False, False], [False, True]])
    ever, eligible = causal_masks(observed)
    np.testing.assert_array_equal(eligible, [[False, False], [False, False], [True, False], [True, False]])
    assert not causal_masks(observed[:1])[0].any()
    assert ever[-1].all()


def test_physical_observations_keep_front_rim_and_ignore_unknown_and_rear():
    arrays = {"offsets": np.array([0, 4]), "vertex_ids": np.array([0, 16, 20, -1])}
    observed = physical_observations(arrays)
    assert np.flatnonzero(observed[0]).tolist() == [0, 16]
    arrays["vertex_ids"] = np.array([0, 16, 16, -1])
    with pytest.raises(ValueError, match="Duplicate"):
        physical_observations(arrays)


def test_summary_reports_missing_predictions_and_true_three_dimensional_rmse():
    result = error_summary(np.array([3., 4., np.nan, 1000.]),
                           np.array([True, True, True, False]),
                           np.array([True, True, False, True]))
    assert result["eligible_count"] == 3
    assert result["scored_count"] == 2 and result["missing_prediction_count"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["mean_mm"] == 3.5
    assert result["rmse_mm"] == pytest.approx(np.sqrt(12.5))


def test_metrics_do_not_silently_drop_uninitialized_surface_predictions():
    gt = np.zeros((2, 20, 3))
    surface = gt.copy()
    surface[1, 0, 0] = .010
    mask = np.zeros((2, 20), dtype=bool)
    mask[1, 0] = True
    arrays = {"gt": gt, "surface_prediction": surface, "online_prediction": gt,
              "eligible": mask, "surface_initialized": np.zeros_like(mask),
              "surface_observed": np.zeros_like(mask),
              "online_available": np.ones_like(mask)}
    result = comparison_metrics(arrays)
    assert result["common_reference"]["surface"]["all"]["mean_mm"] == 10
    assert result["surface_uninitialized_on_reference"] == 1
    assert result["both_initialized"]["surface"]["all"]["eligible_count"] == 0
    surface[0] = 1000
    assert comparison_metrics(arrays) == result


def test_track_binding_follows_real_detection_history_instead_of_nearest_gt():
    graph = SimpleNamespace(vertex_id=torch.tensor([0, 1]))
    episode = SimpleNamespace(graphs=[graph, graph, graph], visible=np.zeros((3, 24), bool))
    collector = PhysicalTrackCollector()
    first = {"detection_rows": torch.tensor([0, 1]), "detection_tracks": torch.tensor([0, 1]),
             "positions": torch.tensor([[0., 0., 0.], [1., 0., 0.]])}
    collector(episode, 0, first)
    # Tracks cross geometrically. A nearest-GT rematch would hide this error.
    crossed = {"detection_rows": torch.tensor([], dtype=torch.long),
               "detection_tracks": torch.tensor([], dtype=torch.long),
               "positions": torch.tensor([[1., 0., 0.], [0., 0., 0.]])}
    collector(episode, 1, crossed)
    assert collector.predictions[-1][0, 0] == 1
    assert collector.predictions[-1][1, 0] == 0
    # A later real detection updates only the evaluator's causal binding.
    switched = {**crossed, "detection_rows": torch.tensor([0]), "detection_tracks": torch.tensor([1])}
    collector(episode, 2, switched)
    assert collector.predictions[-1][0, 0] == 0
    collector(episode, 0, crossed)
    assert not collector.available[-1].any()


def test_input_audit_rejects_changed_gt_surface_tcp_and_non_detector_coordinates():
    common = prepared_episode_arrays()
    slots = {"gt": common["targets"][:, :20].copy(), "tcp": common["tcp"].copy(),
             "detected": np.zeros((3, 20, 3)), "observed": physical_observations(common)}
    surfaces = {"xyz": np.zeros((3, 1, 3)), "valid": np.ones((3, 1), bool)}
    for frame, (start, stop) in enumerate(zip(common["offsets"][:-1], common["offsets"][1:])):
        for row in range(start, stop):
            physical = common["vertex_ids"][row]
            if physical >= 0:
                slots["detected"][frame, physical] = common["xyz"][row]
            else:
                surfaces["xyz"][frame, 0] = common["xyz"][row]
    ever, eligible = causal_masks(slots["observed"])
    saved = {"gt": slots["gt"].copy(), "observed": slots["observed"], "ever_seen": ever, "eligible": eligible}
    audit_episode_inputs(common, slots, surfaces, saved, "fixture")
    slots["gt"][0, 0, 0] += 1
    with pytest.raises(AssertionError):
        audit_episode_inputs(common, slots, surfaces, saved, "fixture")
    slots["gt"] = saved["gt"].copy()
    surfaces["xyz"][0, 0, 0] += 1
    with pytest.raises(AssertionError, match="Surface inputs differ"):
        audit_episode_inputs(common, slots, surfaces, saved, "fixture")
    surfaces["xyz"][0, 0, 0] -= 1
    slots["detected"][0, 0] = [50, 50, 50]
    with pytest.raises(ValueError, match="not a raw detector"):
        audit_episode_inputs(common, slots, surfaces, saved, "fixture")


def test_forbidden_input_is_rejected_before_attempting_to_open_it():
    with pytest.raises(PermissionError, match="Forbidden"):
        checked_file('/does/not/exist/tapnext_state36.npz')


def test_independent_verifier_checks_coordinates_and_rebuilds_episode_masks(tmp_path):
    import hashlib
    import json

    from data_collection.episode_dataset import file_sha256
    from evaluation.corner_tracker_comparison_verification import verify

    observed = np.zeros((3, 20), dtype=bool)
    observed[0, 0] = True
    # A second episode starts at frame 2, so the first episode's sighting must
    # not make its first frame eligible.
    eligible = np.zeros_like(observed)
    eligible[1, 0] = True
    prediction = np.zeros((3, 20, 3))
    prediction[1, 0] = [.003, .004, 0]
    arrays = {
        'gt': np.zeros_like(prediction), 'surface_prediction': prediction,
        'online_prediction': prediction * 2, 'observed': observed,
        'eligible': eligible, 'surface_observed': observed,
        'surface_initialized': np.ones_like(observed), 'online_available': np.ones_like(observed),
        'episode_offsets': np.array([0, 2, 3]), 'sample_ids': np.array(['a:0', 'a:1', 'b:0']),
    }
    path = tmp_path / 'predictions.npz'
    np.savez_compressed(path, **arrays)
    result = {
        'predictions_sha256': file_sha256(path), 'frames': 3,
        'mask_sha256': hashlib.sha256(eligible.tobytes()).hexdigest(),
        'metrics': comparison_metrics(arrays), 'common_eligible_point_frames': 1,
        'weights_unchanged': True,
    }
    (tmp_path / 'results.json').write_text(json.dumps(result))
    verified = verify({'results_root': str(tmp_path)})
    assert verified['status'] == 'passed'
    assert verified['float64_metric_checks'] == 48
    result['metrics']['common_reference']['surface']['all']['mean_mm'] = 999
    (tmp_path / 'results.json').write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        verify({'results_root': str(tmp_path)})


def test_saved_detection_join_recovers_track_order_without_gt_or_nearest_matching():
    from evaluation.corner_tracker_rollout_recovery import detection_tracks

    raw = np.array([[1., 0, 0], [0, 1., 0]], dtype=np.float32)
    output = np.array([[0, 1., 0], [99, 99, 99], [1., 0, 0]], dtype=np.float32)
    joined = detection_tracks(raw, np.array([0, 1]), output, np.array([1, 0, 0]),
                              np.array([True, False, True]))
    assert joined.tolist() == [2, 0]
    with pytest.raises(ValueError, match='Ambiguous duplicate'):
        detection_tracks(np.vstack([raw, raw[:1]]), np.array([0, 1, 0]), output,
                         np.array([1, 0, 0]), np.array([True, False, True]))
    output[2, 0] += 1e-5
    with pytest.raises(ValueError, match='differ from current-frame'):
        detection_tracks(raw, np.array([0, 1]), output, np.array([1, 0, 0]), np.array([True, False, True]))


def test_recovered_trace_preserves_live_online_assignments_and_error_counts(tmp_path):
    import json

    from data_collection.episode_dataset import file_sha256
    from evaluation.corner_tracker_rollout_recovery import recover
    from evaluation.spatial_temporal_corner_tracking import evaluate_online
    from graph.temporal.spatial_temporal_corner_tracker import SpatialTemporalCornerTracker
    from tests.test_spatial_temporal_corner_tracker import prepared_episode

    raw = prepared_episode_arrays()
    episode = prepared_episode()
    np.savez_compressed(tmp_path / 'episode.npz', **raw)
    manifest = {'episodes': [{'file': 'episode.npz', 'sha256': file_sha256(tmp_path / 'episode.npz')}]}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    model = SpatialTemporalCornerTracker(hidden=16, spatial_layers=1, temporal_layers=1,
                                          heads=4, association_dim=8).eval()
    live = PhysicalTrackCollector()
    outputs = []

    def collect(ep, frame, result):
        live(ep, frame, result)
        outputs.append({key: result[key].cpu().numpy().copy()
                        for key in ('positions', 'observed', 'kinds')})

    settings = {'distance_limit_m': .03, 'embedding_weight_m': 0., 'log_every_episodes': 0}
    score = evaluate_online(model, [episode], settings, torch.device('cpu'), frame_callback=collect)
    frame_ids = np.array(['fixture:0', 'fixture:1', 'fixture:2'])
    cache_arrays = {'sample_ids': frame_ids,
                    'offsets': np.r_[0, np.cumsum([len(row['positions']) for row in outputs])],
                    'xyz': np.concatenate([row['positions'] for row in outputs]),
                    'observed': np.concatenate([row['observed'] for row in outputs]),
                    'kinds': np.concatenate([row['kinds'] for row in outputs])}
    association = {'distance_limit_m': .03, 'embedding_weight_m': 0.}
    path = tmp_path / 'rollout.pt'
    torch.save({'format': 'spatial_temporal_test_rollouts_v1', 'split': 'test', 'arrays': cache_arrays,
                'tracker_checkpoint_sha256': 'fixture', 'association': association}, path)
    reference = {'rollout_cache_sha256': file_sha256(path), 'prepared_manifest_sha256': file_sha256(manifest_path),
                 'detector_only_online': score}
    config = {'online_rollout': str(path), 'tracker': {'sha256': 'fixture'}, 'association': association,
              'data': {'manifest': str(manifest_path)}}
    comparison = {'sample_ids': frame_ids, 'observed': physical_observations(raw)}
    recovered, audit = recover(config, comparison, reference)
    assert audit['all_original_integer_counts_preserved']
    assert recovered == score
    np.testing.assert_array_equal(comparison['online_available'], np.array(live.available))
    np.testing.assert_array_equal(comparison['online_prediction'], np.array(live.predictions))
