"""Profiling input validation and CPU-reference orchestration contracts."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import yaml

from data_collection.temporal.corner_tracking_synthetic import synthetic_dataset
from data_collection.temporal.corner_tracks import save_dataset
from tools.benchmark_corner_tracking import benchmark
from tools.corner_tracking_inputs import load_training_episode, require_cuda_device
from tools.profile_corner_eligibility import main as profile_eligibility
from tools.profile_corner_tracking import main as profile_tracking
from training.gnn.temporal.corner_tracking import resolve_config


class CornerTrackingToolTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / 'prepared'
        groups, _ = synthetic_dataset(frames=5, training_episodes=1)
        episodes = [episode for split in groups.values() for episode in split]
        self.manifest = save_dataset(self.data_root, episodes, {})
        self.config = resolve_config({
            'device': 'cpu', 'data': {'kind': 'prepared', 'root': str(self.data_root)},
            'architectures': ['transformer'], 'seeds': [0],
            'model': {'d_model': 16, 'd_state': 4, 'd_conv': 2},
            'gnn': {'hidden': 16, 'layers': 1, 'global_geometry': False, 'readout': 'attention'},
        })

    def write_manifest(self):
        (self.data_root / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_loader_selects_training_episode_without_truncating_frames(self):
        self.manifest['episodes'].reverse()
        self.write_manifest()
        episode, manifest = load_training_episode(self.data_root)
        training_row = next(row for row in self.manifest['episodes'] if row['split'] == 'train')
        self.assertEqual(episode.name, training_row['episode'])
        self.assertEqual(len(episode.frames), 5)
        self.assertEqual(manifest['dataset_sha256'], self.manifest['dataset_sha256'])
        self.assertEqual(episode.frames[0].observation.xyz.dtype.name, 'float32')

    def test_loader_rejects_corruption_before_reading_episode(self):
        row = self.manifest['episodes'][0]
        (self.data_root / row['file']).write_bytes(b'corrupt')
        with patch('tools.corner_tracking_inputs.read_episode') as reader:
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_training_episode(self.data_root)
        reader.assert_not_called()

    def test_loader_rejects_paths_outside_dataset(self):
        row = self.manifest['episodes'][0]
        external = self.root / 'external.npz'
        external.write_bytes((self.data_root / row['file']).read_bytes())
        row['file'] = '../external.npz'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'must stay inside'):
            load_training_episode(self.data_root)

    def test_loader_explains_missing_training_split(self):
        for row in self.manifest['episodes']:
            row['split'] = 'validation'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'no training episode'):
            load_training_episode(self.data_root)

    def test_gpu_tools_reject_cpu_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, 'requires a CUDA device'):
            require_cuda_device('cpu')
        with patch('tools.benchmark_corner_tracking.load_training_episode') as loader:
            with self.assertRaisesRegex(ValueError, 'requires a CUDA device'):
                benchmark(self.config, 'joint', 3, 1)
        loader.assert_not_called()

        recipe = self.root / 'profile.yaml'
        recipe.write_text(yaml.safe_dump(self.config))
        output = self.root / 'profile'
        arguments = ['profile', '--config', str(recipe), '--architecture', 'transformer', '--output', str(output)]
        with patch.object(sys, 'argv', arguments), patch('tools.profile_corner_tracking.load_training_episode') as loader:
            with self.assertRaisesRegex(ValueError, 'requires a CUDA device'):
                profile_tracking()
        loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_unavailable_cuda_uses_existing_device_validation(self):
        with patch('utils.device_set.torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA device cuda:0 is not available'):
                require_cuda_device('cuda:0')

    def test_benchmark_repeats_restore_initial_weights(self):
        # Exercise actual Transformer/loss/AdamW on CPU; CUDA allocation/timing
        # calls are mocked, so this is not a CUDA execution or timing result.
        with patch('tools.benchmark_corner_tracking.require_cuda_device', return_value=torch.device('cpu')), \
             patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), \
             patch('torch.cuda.max_memory_allocated', return_value=0):
            report = benchmark(self.config, 'joint', frames=5, repeats=1)
        cold, warm = report['measurements']
        self.assertTrue(cold['cold'])
        self.assertFalse(warm['cold'])
        for key in ('loss', 'gradient_norm', 'supervised_nodes'):
            self.assertEqual(cold[key], warm[key])
        self.assertGreater(cold['supervised_nodes'], 0)
        self.assertEqual(report['frames'], 5)
        self.assertGreaterEqual(report['median_warm_seconds'], 0)

    def test_cpu_eligibility_profile_preserves_repeated_result_hash(self):
        output = self.root / 'eligibility.json'
        arguments = ['eligibility', '--data-root', str(self.data_root), '--output', str(output)]
        with patch.object(sys, 'argv', arguments):
            profile_eligibility()
        first = json.loads(output.read_text())
        with patch.object(sys, 'argv', arguments):
            profile_eligibility()
        second = json.loads(output.read_text())
        self.assertEqual(first['eligibility_sha256'], second['eligibility_sha256'])
        self.assertEqual(first['dataset_sha256'], second['dataset_sha256'])
        self.assertEqual(first['frames'], 5)
        self.assertEqual(len(first['seconds']), 2)
        self.assertTrue(all(value >= 0 for value in first['seconds']))


if __name__ == '__main__':
    unittest.main()
