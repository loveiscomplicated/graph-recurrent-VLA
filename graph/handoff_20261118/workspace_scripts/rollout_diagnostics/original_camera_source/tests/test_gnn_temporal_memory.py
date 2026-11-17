"""Causal episode replay, diagnostic isolation and evaluation entrypoint contracts."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluation.gnn.temporal_memory import replay_memory, sample_order, scoring_groups
from graph.utils.hole_memory import MemorySettings


class TemporalMemoryEvaluationTests(unittest.TestCase):
    def test_prefix_is_independent_of_future_frames_and_episodes_reset(self):
        prediction = np.zeros((5, 21))
        prediction[:, 10] = [0.1, 0.2, 0.3, 8, 9]
        quality = np.array([[6, .9, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [6, .9, 1]])
        tcp = np.zeros((5, 3))
        episodes = np.array(['a', 'a', 'a', 'b', 'b'])
        frames = np.array([0, 1, 2, 0, 1])
        settings = MemorySettings(consecutive_frames=1)
        full = replay_memory(prediction, tcp, quality, episodes, frames, settings)
        prefix = replay_memory(prediction[:2], tcp[:2], quality[:2], episodes[:2], frames[:2], settings)
        np.testing.assert_array_equal(full['object21'][:2], prefix['object21'])
        self.assertEqual(full['object21'][2, 10], .1)
        self.assertFalse(full['initialized'][3])
        self.assertEqual(full['object21'][3, 10], 8)

        self.assertEqual(sample_order('shard_000001/episode_000002/frame_000010.npz'),
                         ('shard_000001/episode_000002', 10))

    def test_diagnostic_labels_are_aligned_by_id_and_never_enter_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.npz'
            front = np.array([[True] * 4 + [False] * 4] * 2)
            distances = np.full((2, 8), np.inf)
            distances[0, :4] = 0
            np.savez(path, sample_ids=np.array(['a/frame_000001.npz', 'a/frame_000000.npz']),
                     front=front, min_distance_m=distances, foreground=np.ones((2, 8), dtype=bool))
            data = {'sample_ids': np.array(['a/frame_000000.npz', 'a/frame_000001.npz']),
                    'quality': np.zeros((2, 3))}
            groups = scoring_groups(data, path)
            np.testing.assert_array_equal(groups['strong_occlusion_proxy'], [True, False])


    def test_main_dispatch_records_an_evaluation_without_training(self):
        from types import SimpleNamespace
        from data_collection.episode_dataset import file_sha256
        from main import load_config, run
        from training.gnn.trainer import resolve_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(1, 1)
            training_config = resolve_config(load_config('graph/configs/train_gated_mpnn_cached.yaml'))
            checkpoint = root / 'best.pt'
            torch.save({'kind': 'best_weights', 'model': model.state_dict(),
                        'config': training_config, 'dataset_sha256': 'synthetic'}, checkpoint)
            original_hash = file_sha256(checkpoint)
            config = {
                'trainer': 'evaluation.gnn.temporal_memory:evaluate', 'device': 'cpu',
                'checkpoint': str(checkpoint), 'data': {'cache_dir': str(root)},
                'evaluation': {'visibility_labels': None},
                'memory': {'quality_gated': asdict(MemorySettings(consecutive_frames=1))},
                'output': {'root': str(root / 'outputs')},
            }
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            prediction = np.zeros((2, 21), dtype=np.float32)
            prediction[:, 10] = [.1, .2]
            target = np.zeros_like(prediction)
            target[:, 10] = .1
            data = {'sample_ids': np.array(['a/frame_000000.npz', 'a/frame_000001.npz']),
                    'episode': np.array(['a', 'a']), 'frame': np.array([0, 1]),
                    'prediction': prediction, 'target': target, 'tcp_world': np.zeros((2, 3)),
                    'quality': np.array([[6, .9, 1], [0, 0, 0]])}
            module = 'evaluation.gnn.temporal_memory'
            bundle = SimpleNamespace(identity={'dataset_sha256': 'synthetic'}, validation=None)
            with patch.dict('os.environ', {'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(root/'experiments')}), \
                 patch(module+'.build_data', return_value=bundle), \
                 patch(module+'.build_model', return_value=model), \
                 patch(module+'.infer_validation', return_value=data), \
                 patch('builtins.print'):
                run(path)
            output = next((root/'outputs').iterdir())
            summary = json.loads((output/'summary.json').read_text())
            self.assertAlmostEqual(summary['methods']['quality_gated']['groups']['all']['mean_mm'], 0)
            self.assertGreater(summary['methods']['gnn']['groups']['all']['mean_mm'], 0)
            self.assertEqual(file_sha256(checkpoint), original_hash)
            metadata = json.loads(next((root/'experiments').glob('*/metadata.json')).read_text())
            self.assertEqual(metadata['status'], 'finished')
            self.assertFalse(metadata['memory_uses_gt'])



if __name__ == '__main__':
    unittest.main()
