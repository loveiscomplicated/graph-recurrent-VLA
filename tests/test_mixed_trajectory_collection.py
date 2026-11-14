"""Checks for reproducible split and mixed-action collection contracts."""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256


class MixedTrajectoryProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Defer ManiSkill/imageio imports so logger warning tests remain isolated.
        from data_collection import mixed_trajectory_collection
        cls.collection = mixed_trajectory_collection

    def setUp(self):
        self.config = dict(seed_generator=20261114, seed_range=[3_000_000, 4_000_000],
                           pilot_count=100, split_counts=dict(train=700, validation=150, test=150),
                           random_limits=[0.10, 0.25, 0.50], camera={'eye': [0, 0, 0]},
                           control_mode='pd_ee_delta_pose', max_steps=250, visibility_stride=25)

    @patch('data_collection.mixed_trajectory_collection.load_exclusions', return_value=({3000010}, {'source': 'hash'}))
    def test_protocol_is_reproducible_disjoint_and_balanced(self, _load):
        first = self.collection.build_protocol(self.config)
        self.assertEqual(first, self.collection.build_protocol(self.config))
        records = first['records']
        self.assertEqual(len(records), 1100)
        self.assertEqual(len({row['seed'] for row in records}), 1100)
        self.assertNotIn(3000010, {row['seed'] for row in records})
        self.assertEqual({name: sum(row['split'] == name for row in records)
                          for name in ('pilot', 'train', 'validation', 'test')},
                         dict(pilot=100, train=700, validation=150, test=150))
        self.assertEqual(set(first['modes']), set(self.collection.MODES))
        for split in ('pilot', 'train', 'validation', 'test'):
            rows = [row for row in records if row['split'] == split]
            self.assertEqual({row['transition_phase'] for row in rows}, set(self.collection.PHASES))
            self.assertEqual({row['random_gripper'] for row in rows}, {'hold', 'random'})
            self.assertTrue(all(0 <= row['transition_delay'] < 10 for row in rows))
            groups = [(row['transition_phase'], row['random_limit'], row['random_gripper']) for row in rows]
            counts = [groups.count(group) for group in set(groups)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_random_action_uses_native_shape_and_bounds(self):
        class ActionSpace:
            shape = (7,)
        class Task:
            device = torch.device('cpu')
            action_space = ActionSpace()
        action = self.collection.random_action(np.random.default_rng(10), 0.25, -1.0, Task())
        self.assertEqual(tuple(action.shape), (7,))
        self.assertEqual(action[-1].item(), -1.0)
        self.assertTrue(torch.all(torch.abs(action[:6]) <= 0.25))

    def test_main_requires_a_matching_audit_and_sufficient_space(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = {'records': ['fixed']}
            protocol_path = root / 'protocol.json'
            protocol_path.write_text(json.dumps(protocol))
            audit_path = root / 'audit.json'
            audit_path.write_text(json.dumps(dict(protocol_sha256=file_sha256(protocol_path),
                                                  initial_states=1, episodes=3,
                                                  estimated_main_bytes=100)))
            config = dict(collect_splits=['train'], pilot_audit=str(audit_path),
                          pilot_root=str(root), pilot_count=1,
                          output_root=str(root / 'main'), minimum_free_gib=5)
            with patch('data_collection.mixed_trajectory_collection.shutil.disk_usage') as usage:
                usage.return_value.free = 1
                with self.assertRaisesRegex(RuntimeError, 'Insufficient storage'):
                    self.collection.check_main_preconditions(config, protocol)
                usage.return_value.free = 100 * 1024**3
                self.collection.check_main_preconditions(config, protocol)
                with self.assertRaisesRegex(ValueError, 'protocol differs'):
                    self.collection.check_main_preconditions(config, {'records': ['changed']})


if __name__ == '__main__':
    unittest.main()
