"""Temporal coordinates, causality, parent isolation and short CPU workflow smoke."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal_gate_data import make_gate_manifest, validate_gate_splits
from evaluation.gnn.temporal_gate import evaluate, replay_gate
from graph.build_graph import GraphBuilder
from graph.temporal_gate import (
    LearnedTemporalGate, correct_object21, corrected_state38, observation_features, rollout,
)
from main import load_config
from training.gnn.temporal_gate import (
    infer_observations, load_frozen_source, run_gate_epoch, train,
)
from training.gnn.trainer import build_model, resolve_config
from training.gnn.data import TrainingBatch


def synthetic_data(episodes=('a', 'b'), length=4):
    count = len(episodes) * length
    prediction = np.zeros((count, 21), dtype=np.float32)
    prediction[:, 10] = np.tile([.1, .18, .25, .12][:length], len(episodes))
    target = np.zeros_like(prediction)
    target[:, 10] = .1
    names = np.repeat(episodes, length)
    frames = np.tile(np.arange(length), len(episodes))
    return {'prediction': prediction, 'target': target,
            'tcp_world': np.zeros((count, 3), dtype=np.float32),
            'features': np.zeros((count, 17), dtype=np.float32),
            'quality': np.tile([[6, .9, 1], [0, 0, 0], [0, 0, 0], [6, .9, 1]][:length], (len(episodes), 1)),
            'episode': names, 'frame': frames,
            'sample_ids': np.asarray([f'{ep}/frame_{frame:06d}.npz' for ep, frame in zip(names, frames)])}


class TemporalGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.gate = LearnedTemporalGate()
        # A known scalar alpha=.5 makes coordinate and autoregression checks exact.
        for parameter in self.gate.parameters():
            torch.nn.init.zeros_(parameter)

    def test_world_coordinates_follow_moving_tcp_and_cold_start_has_no_gt(self):
        quality = torch.zeros(17)
        first, alpha = self.gate.step(torch.tensor([.1, .2, .3]), torch.tensor([1., 0, 0]), quality)
        torch.testing.assert_close(first, torch.tensor([1.1, .2, .3]))
        self.assertEqual(alpha.item(), 1)
        second, alpha = self.gate.step(torch.tensor([-.9, .2, .3]), torch.tensor([2., 0, 0]), quality, first)
        torch.testing.assert_close(second, first)
        self.assertEqual(alpha.item(), .5)

    def test_autoregressive_memory_prefix_and_reset(self):
        data = synthetic_data()
        data['prediction'][4:, 10] += 10
        result = replay_gate(self.gate, data, 'cpu')
        np.testing.assert_allclose(result['world'][:3, 0], [.1, .14, .195], atol=1e-7)
        self.assertAlmostEqual(result['world'][4, 0], 10.1, places=5)
        prefix = {key: value[:2] for key, value in data.items()}
        short = replay_gate(self.gate, prefix, 'cpu')
        np.testing.assert_array_equal(short['world'], result['world'][:2])
        # Labels, including previous GT and future GT, cannot affect rollout.
        data['target'][:] = 1000
        np.testing.assert_array_equal(replay_gate(self.gate, data, 'cpu')['world'], result['world'])
        data['prediction'][3, 10] = 1000
        np.testing.assert_array_equal(replay_gate(self.gate, data, 'cpu')['world'][:3], result['world'][:3])

    def test_recurrence_backpropagates_through_own_previous_memory(self):
        data = synthetic_data(episodes=('a',))
        prediction = torch.tensor(data['prediction'], requires_grad=True)
        world, alpha = rollout(self.gate, prediction, torch.tensor(data['tcp_world']), torch.tensor(data['features']))
        world[-1].sum().backward()
        self.assertAlmostEqual(prediction.grad[0, 10].item(), .125)
        self.assertTrue(torch.all((alpha >= 0) & (alpha <= 1)))
        self.assertEqual(sum(p.numel() for p in self.gate.parameters()), 2081)

    def test_schema_quality_masks_padding_and_unknown_confidence(self):
        graph = SimpleNamespace(node_valid=torch.tensor([[True, True, True, False]]),
                                object_type=torch.tensor([[1, 1, 2, -1]]),
                                confidence_valid=torch.tensor([[True, False, False, False]]),
                                confidence=torch.tensor([[.8, 0, 0, 0]]))
        features = observation_features(graph)[0]
        torch.testing.assert_close(features[:8], torch.tensor([2, 1, 0, .5, 1, .8, .8, .8]))
        torch.testing.assert_close(features[8:16], torch.tensor([1., 1, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(features[-1], 0)
        graph.node_valid[:] = False
        features = observation_features(graph)[0]
        self.assertEqual(features[2], 1)
        self.assertEqual(features[10], 1)
        self.assertEqual(features[-1], 1)
        self.assertTrue(torch.isfinite(features).all())

    def test_object21_and_state38_preserve_tip_and_other_fields(self):
        original = torch.arange(42, dtype=torch.float32).reshape(2, 21)
        hole = torch.ones(2, 3)
        corrected = correct_object21(original, hole)
        torch.testing.assert_close(corrected[:, 7:10] + corrected[:, 10:13],
                                   original[:, 7:10] + original[:, 10:13])
        unchanged = list(range(7)) + list(range(13, 21))
        torch.testing.assert_close(corrected[:, unchanged], original[:, unchanged])
        hole_only = correct_object21(original, hole, False)
        torch.testing.assert_close(hole_only[:, 7:10], original[:, 7:10])
        state = corrected_state38(torch.zeros(2, 15), original, torch.ones(2, 2), hole)
        self.assertEqual(state.shape, (2, 38))
        torch.testing.assert_close(state[:, 15:36], corrected)

    def test_parent_and_gnn_validation_leakage_rejected(self):
        source = {'train': ['a', 'b'], 'validation': ['v'], 'test': ['t']}
        manifest = {'splits': {'train': ['a'], 'validation': ['b'], 'test': ['v']},
                    'parent_by_episode': {name: name for name in ('a', 'b', 'v', 't')}}
        validate_gate_splits(manifest, source)
        for kind in ('gnn_validation', 'sibling', 'reserved_sibling', 'unknown_parent'):
            broken = deepcopy(manifest)
            if kind == 'gnn_validation':
                broken['splits']['train'] = ['v']
            elif kind == 'sibling':
                broken['parent_by_episode']['b'] = 'a'
            elif kind == 'reserved_sibling':
                broken['parent_by_episode']['t'] = 'a'
            else:
                del broken['parent_by_episode']['a']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_gate_splits(broken, source)

    def test_manifest_creation_is_parent_disjoint_and_deterministic(self):
        source = {'train': ['a', 'a_sibling', 'b', 'reserved_sibling'],
                  'validation': ['v'], 'test': ['t']}
        parents = {'a': 'p', 'a_sibling': 'p', 'b': 'q', 'reserved_sibling': 'r',
                   'v': 'r', 't': 's'}
        manifest = make_gate_manifest(source, parents)
        self.assertEqual(manifest, make_gate_manifest(source, parents))
        self.assertNotIn('reserved_sibling', manifest['splits']['train'] + manifest['splits']['validation'])
        for split in ('train', 'validation'):
            if 'a' in manifest['splits'][split]:
                self.assertIn('a_sibling', manifest['splits'][split])

    def test_real_tiny_cache_end_to_end_smoke(self):
        from main import run
        from training.gnn.cached_data import build_data
        from training.gnn.data import canonical_hash, save_cached_sample

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            (cache / 'metadata').mkdir(parents=True)
            source_config = resolve_config(load_config('graph/configs/train_gated_mpnn_cached.yaml'))
            source_config['model'].update(hidden=8, layers=1, edge_embedding_dim=4)
            source_config['data']['cache_dir'] = str(cache)
            metadata = {'preprocessing': {'format_version': 1, 'camera': 'base_camera',
                        'graph': source_config['graph'], 'confidence': source_config['confidence']},
                        'splits': {'train': ['a', 'b'], 'validation': ['v'], 'test': ['t']}, 'frames': []}
            namespace = canonical_hash(metadata['preprocessing'])
            inventory = {'preprocessing_sha256': namespace, 'files': []}
            observation = SimpleNamespace(xyz=torch.zeros(1, 3), object_id=torch.tensor([1]), metadata=[{}])
            graph = GraphBuilder()([observation], torch.zeros(1, 3))
            target = torch.zeros(1, 21)
            target[:, 10] = .1
            for episode in ('a', 'b', 'v'):
                for frame in range(3):
                    sample_id = f'{episode}/frame_{frame:06d}.npz'
                    digest = canonical_hash({'id': sample_id})
                    metadata['frames'].append({'id': sample_id, 'sha256': digest})
                    identity = {'frame_sha256': digest, 'sample_id': sample_id, 'preprocessing_sha256': namespace}
                    relative = f'{namespace}/{canonical_hash(identity)}.npz'
                    path = cache / relative
                    save_cached_sample(path, TrainingBatch(graph, {'object21': target}, [sample_id]), identity)
                    inventory['files'].append({'sample_id': sample_id, 'file': relative,
                                               'sha256': file_sha256(path), 'size': path.stat().st_size})
            metadata['dataset_sha256'] = canonical_hash(metadata)
            metadata['sample_counts'] = {'train': 6, 'validation': 3, 'test': 0}
            (cache / 'metadata/data_identity.json').write_text(json.dumps(metadata))
            (cache / 'metadata/cache_inventory.json').write_text(json.dumps(inventory))
            bundle = build_data(source_config, torch.Generator().manual_seed(0))
            source = root / 'source.pt'
            frozen = build_model(source_config['model'], torch.device('cpu'))
            torch.save({'kind': 'best_weights', 'model': frozen.state_dict(), 'config': source_config,
                        'dataset_sha256': bundle.identity['dataset_sha256']}, source)
            manifest = root / 'splits.json'
            manifest.write_text(json.dumps(make_gate_manifest(metadata['splits'], {name: name for name in 'abvt'})))
            config = {'trainer': 'training.gnn.temporal_gate:train', 'device': 'cpu',
                      'checkpoint': str(source), 'split_manifest': str(manifest),
                      'data': {'cache_dir': str(cache)}, 'training': {'epochs': 1},
                      'rule_memory': {}, 'output': {'root': str(root / 'train')}}
            config_path = root / 'config.json'
            original = file_sha256(source)
            with patch.dict('os.environ', {'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(root / 'experiments')}):
                config_path.write_text(json.dumps(config))
                run(config_path)
                best = next((root / 'train').glob('*/best.pt'))
                config.update(trainer='evaluation.gnn.temporal_gate:evaluate', gate_checkpoint=str(best))
                config['output']['root'] = str(root / 'evaluation')
                config_path.write_text(json.dumps(config))
                run(config_path)
            summary = json.loads(next((root / 'evaluation').glob('*/summary.json')).read_text())
            self.assertEqual(summary['frames'], 3)
            self.assertEqual(summary['episodes'], 1)
            self.assertEqual(original, file_sha256(source))

    def test_frozen_vertex_model_and_cpu_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            config = resolve_config(load_config('graph/configs/train_gated_mpnn_cached.yaml'))
            config['model'].update(hidden=8, layers=1, edge_embedding_dim=4)
            original = build_model(config['model'], torch.device('cpu'))
            path = Path(directory) / 'best.pt'
            torch.save({'kind': 'best_weights', 'config': config, 'model': original.state_dict(),
                        'dataset_sha256': 'fixture'}, path)
            digest = file_sha256(path)
            frozen, _, _ = load_frozen_source({'checkpoint': str(path), 'data': {'cache_dir': directory},
                                               'evaluation': {'batch_size': 2}}, torch.device('cpu'))
            self.assertFalse(frozen.training)
            self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
            observation = SimpleNamespace(xyz=torch.zeros(1, 3), object_id=torch.tensor([1]), metadata=[{}])
            graph = GraphBuilder()([observation], torch.zeros(1, 3))
            batch = TrainingBatch(graph, {'object21': torch.zeros(1, 21)}, ['a/frame_000000.npz'])
            data = infer_observations(frozen, [batch], 'cpu')
            self.assertEqual(data['prediction'].shape, (1, 21))
            optimizer = torch.optim.Adam(self.gate.parameters(), lr=.001)
            run_gate_epoch(self.gate, data, 'cpu', {'loss': 'l1', 'gradient_clip_norm': 1},
                           optimizer, np.random.default_rng(0))
            self.assertTrue(all(p.grad is None for p in frozen.parameters()))
            self.assertEqual(digest, file_sha256(path))
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, frozen.state_dict()[name])

    def test_short_train_evaluate_workflow_smoke(self):
        # Synthetic replay isolates the new workflow; real cache indexing is tested separately.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.pt'
            source.write_bytes(b'unchanged fixture')
            manifest = root / 'splits.json'
            manifest.write_text('{}')
            config = {'checkpoint': str(source), 'split_manifest': str(manifest), 'data': {'cache_dir': directory},
                      'device': 'cpu', 'training': {'epochs': 1}, 'rule_memory': {},
                      'output': {'root': str(root / 'training')}}
            data = synthetic_data()
            def prepare_fixture(effective, device, requested):
                effective.update(source_checkpoint_sha256=file_sha256(source),
                                 split_manifest_sha256=file_sha256(manifest),
                                 source_training_config={}, feature_names=['fixture'])
                return torch.nn.Linear(1, 1).requires_grad_(False), dict.fromkeys(requested), {'dataset_sha256': 'fixture'}
            before = file_sha256(source)
            with patch.dict('os.environ', {'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(root / 'experiments')}), \
                 patch('training.gnn.temporal_gate.prepare', side_effect=prepare_fixture), \
                 patch('training.gnn.temporal_gate.infer_observations', return_value=data):
                trained = train(config)
            config['gate_checkpoint'] = str(trained / 'best.pt')
            config['output']['root'] = str(root / 'evaluation')
            with patch.dict('os.environ', {'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(root / 'experiments')}), \
                 patch('evaluation.gnn.temporal_gate.prepare', side_effect=prepare_fixture), \
                 patch('evaluation.gnn.temporal_gate.infer_observations', return_value=data):
                output = evaluate(config)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(set(summary['methods']), {'gnn', 'rule', 'learned'})
            self.assertEqual(summary['frames'], 8)
            self.assertTrue((output / 'episode_trace_000.png').is_file())
            self.assertEqual(before, file_sha256(source))
            self.assertEqual(len(list((root / 'experiments').iterdir())), 2)


if __name__ == '__main__':
    unittest.main()
