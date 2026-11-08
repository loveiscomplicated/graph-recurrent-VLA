"""Verify that the read-only audit reproduces the training metric denominators."""

from copy import deepcopy
import unittest

import numpy as np
import torch

from data_collection.temporal.corner_tracking_synthetic import synthetic_dataset
from evaluation.gnn.temporal.corner_tracking_diagnostics import AuditTracker, audit_episode, merge_reports
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.corner_tracking.model import ConstantVelocityMotion, CornerMotionModel
from graph.temporal.corner_tracking.tracker import TrackingSettings
from tests.fixtures.corner_tracking import calibration, observation
from training.gnn.temporal.corner_tracking_rollout import replay_episode


class TrackingDiagnosticTests(unittest.TestCase):
    def test_absolute_cost_does_not_discount_distant_dormant_tracks(self):
        gate = calibration(radius=.05)
        gate.radii[:, 1] = .002
        tracker = AuditTracker(ConstantVelocityMotion(), gate)
        points = [[.01, 0, 0], [.001, 0, 0]]
        tracker.step(observation(points))
        tracks = list(tracker.tracks.values())
        tracks[0].anchor_frame = 0
        tracks[1].anchor_frame = 3
        tracker.frame = 4
        positions = np.asarray(points, dtype=np.float32)
        detection = observation([[0, 0, 0]])
        baseline = tracker._associate(tracks, positions, detection)[0]
        tracker.association = 'absolute_distance'
        revised = tracker._associate(tracks, positions, detection)[0]
        self.assertEqual(baseline, {0: 0})
        self.assertEqual(revised, {1: 0})

    def test_baseline_matches_replay_counters_and_partitions_missing_frames(self):
        torch.set_num_threads(1)
        groups, _ = synthetic_dataset(frames=18, training_episodes=1)
        episode = groups['train'][0]
        plan = dict(mode='single', start=4, stop=12, references=[0], kind=0)
        motion = CornerMotionModel(d_model=16, d_state=4, d_conv=2)
        gnn = VertexEdgeGatedMPNN(16, hidden=16, layers=1, extra_node_dim=5)
        gate = calibration()
        gate.variance_scale[:] = .1
        with torch.no_grad():
            replay = replay_episode(motion, gnn, episode, plan, gate, TrackingSettings())
            audit = audit_episode(motion, episode, plan, gate, TrackingSettings(), 'baseline')
        for original, key in (('id_switches', 'id_switches'),
                              ('track_identity_changes', 'track_identity_changes'),
                              ('reobservations', 'reobservations'),
                              ('false_reconnections', 'false_reconnections'),
                              ('duplicate_tracks', 'duplicate_track_frames')):
            self.assertEqual(replay['counters'][original], audit['counts'].get(key, 0))
        current = audit['admission']['current']
        self.assertEqual(replay['counters']['certified_missing_corner_frames'], current['missing_corner_frames'])
        self.assertEqual(replay['counters']['covered_missing_corner_frames'], current['covered_missing_corner_frames'])
        rejected = sum(value for key, value in audit['counts'].items() if key.startswith('missing_excluded:'))
        self.assertEqual(rejected + current['covered_missing_corner_frames'], current['missing_corner_frames'])
        combined = merge_reports([audit, audit])
        self.assertEqual(combined['admission']['current']['missing_corner_frames'], 2 * current['missing_corner_frames'])
        for probe in combined['admission'].values():
            self.assertLessEqual(probe['accurate_coverage_5mm'], probe['coverage'])
            self.assertLessEqual(probe['adoption_precision_5mm'], 1)

    def test_gt_coordinates_only_change_audit_errors(self):
        torch.set_num_threads(1)
        groups, _ = synthetic_dataset(frames=12, training_episodes=1)
        episode = groups['train'][0]
        changed = deepcopy(episode)
        for frame in changed.frames:
            frame.reference_xyz += 10
        plan = dict(mode='single', start=4, stop=10, references=[0], kind=0)
        motion = CornerMotionModel(d_model=16, d_state=4, d_conv=2)
        gate = calibration()
        gate.variance_scale[:] = .1
        with torch.no_grad():
            baseline = audit_episode(motion, episode, plan, gate, TrackingSettings(), 'recent_first')
            alternative = audit_episode(motion, changed, plan, gate, TrackingSettings(), 'recent_first')
        self.assertEqual(baseline['counts'], alternative['counts'])
        self.assertEqual(baseline['switches'], alternative['switches'])
        for name in baseline['admission']:
            self.assertEqual(baseline['admission'][name]['covered_missing_corner_frames'],
                             alternative['admission'][name]['covered_missing_corner_frames'])
            self.assertGreater(sum(baseline['errors'].get('adopted_predictions', [])), 0)
            self.assertGreater(sum(alternative['errors'].get('adopted_predictions', [])),
                               sum(baseline['errors'].get('adopted_predictions', [])))

    def test_known_identity_errors_are_counted_in_adoption_precision(self):
        torch.set_num_threads(1)
        groups, _ = synthetic_dataset(frames=12, training_episodes=1)
        episode = deepcopy(groups['train'][0])
        # Change only offline labels to create a known identity disagreement,
        # then hide all observations so those tracks can emit predictions.
        for frame in episode.frames[3:6]:
            frame.physical_ids[:2] = frame.physical_ids[:2][::-1]
        frame = episode.frames[6]
        keep = np.zeros(len(frame.physical_ids), dtype=bool)
        frame.observation = frame.observation.subset(keep)
        frame.physical_ids = frame.physical_ids[keep]
        motion = CornerMotionModel(d_model=16, d_state=4, d_conv=2)
        gate = calibration()
        gate.variance_scale[:] = .1
        with torch.no_grad():
            report = audit_episode(motion, episode, dict(mode='natural', start=0, stop=0),
                                   gate, TrackingSettings(), 'baseline')
        current = report['admission']['current']
        self.assertGreater(current['inconsistent_adoptions'], 0)
        self.assertGreaterEqual(current['auditable_adoptions'] - current['accurate_adoptions_5mm'],
                                current['inconsistent_adoptions'])


if __name__ == '__main__':
    unittest.main()
