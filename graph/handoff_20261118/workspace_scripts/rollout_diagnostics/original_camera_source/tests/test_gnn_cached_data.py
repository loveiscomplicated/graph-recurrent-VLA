"""Cache-only indexing and integrity contracts, without detection or training."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from data_collection.episode_dataset import file_sha256
from graph.build_graph import GraphBuilder
from main import load_config, run
from training.gnn.cached_data import build_data
from training.gnn.data import TrainingBatch, canonical_hash
from training.gnn.trainer import resolve_config


class CachedDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'metadata').mkdir()
        self.config = resolve_config(load_config('graph/configs/train_gated_mpnn_cached.yaml'))
        self.config['data']['cache_dir'] = str(self.root)
        self.metadata = {
            'preprocessing': {'format_version': 1, 'camera': 'base_camera',
                              'graph': self.config['graph'], 'confidence': self.config['confidence']},
            'splits': {'train': ['episode_a'], 'validation': ['episode_b'], 'test': ['episode_c']},
            'frames': [{'id': 'episode_a/frame_000000.npz', 'sha256': 'a' * 64},
                       {'id': 'episode_b/frame_000000.npz', 'sha256': 'b' * 64}],
        }
        namespace = canonical_hash(self.metadata['preprocessing'])
        (self.root / namespace).mkdir()
        self.inventory = {'preprocessing_sha256': namespace, 'files': []}
        for index, frame in enumerate(self.metadata['frames']):
            expected = {'frame_sha256': frame['sha256'], 'sample_id': frame['id'],
                        'preprocessing_sha256': namespace}
            relative = f'{namespace}/{canonical_hash(expected)}.npz'
            path = self.root / relative
            # Not real cache material: schema loading is mocked in these tests.
            path.write_bytes(f'placeholder-{index}'.encode())
            self.inventory['files'].append({'sample_id': frame['id'], 'file': relative,
                                           'sha256': file_sha256(path), 'size': path.stat().st_size})
        self.write_metadata()
        observation = SimpleNamespace(xyz=torch.zeros(1, 3), object_id=torch.tensor([0]), metadata=[{}])
        graph = GraphBuilder()([observation], torch.zeros(1, 3))
        self.sample = TrainingBatch(graph, {'object21': torch.zeros(1, 21)}, ['placeholder'])

    def write_metadata(self):
        original = {key: value for key, value in self.metadata.items()
                    if key not in ('dataset_sha256', 'sample_counts')}
        self.metadata['dataset_sha256'] = canonical_hash(original)
        self.metadata['sample_counts'] = {'train': 1, 'validation': 1, 'test': 0}
        (self.root / 'metadata/data_identity.json').write_text(json.dumps(self.metadata))
        (self.root / 'metadata/cache_inventory.json').write_text(json.dumps(self.inventory))

    def bundle(self, config=None):
        return build_data(config or self.config, torch.Generator().manual_seed(0))

    def test_loads_saved_splits_without_raw_data_detector_or_cache_writes(self):
        def load(path, identity, sample_id):
            sample = deepcopy(self.sample)
            sample.sample_ids = [sample_id]
            return sample

        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob('*') if path.is_file()}
        with patch('training.gnn.data.read_episodes', side_effect=AssertionError('raw access')), \
             patch('training.gnn.data.preprocessing_identity', side_effect=AssertionError('detector access')), \
             patch('training.gnn.data.save_cached_sample', side_effect=AssertionError('cache write')), \
             patch('training.gnn.cached_data.load_cached_sample', side_effect=load):
            bundle = self.bundle()
            self.assertEqual(next(iter(bundle.train)).sample_ids, ['episode_a/frame_000000.npz'])
            self.assertEqual(next(iter(bundle.validation)).sample_ids, ['episode_b/frame_000000.npz'])
            self.assertTrue(bundle.identity['cache_only'])
            self.assertEqual(len(bundle.train.dataset), 1)
        after = {str(path.relative_to(self.root)): path.read_bytes()
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_detects_same_size_corruption_on_read(self):
        bundle = self.bundle()
        path = bundle.train.dataset.samples[0].path
        data = path.read_bytes()
        path.write_bytes(b'X' + data[1:])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            bundle.train.dataset[0]

    def test_missing_cache_fails_without_regeneration(self):
        (self.root / self.inventory['files'][0]['file']).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing cached frame'):
            self.bundle()

    def test_metadata_checksum_is_required(self):
        self.metadata['dataset_sha256'] = '0' * 64
        (self.root / 'metadata/data_identity.json').write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, 'data_identity checksum'):
            self.bundle()

    def test_rejects_overlap_and_reserved_test_frames(self):
        self.metadata['splits']['test'].append('episode_a')
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.bundle()
        self.metadata['splits']['test'] = ['episode_c', 'episode_a']
        self.metadata['splits']['train'] = ['episode_d']
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'reserved test'):
            self.bundle()

    def test_rejects_wrong_namespace_inventory_and_duplicates(self):
        self.inventory['preprocessing_sha256'] = '0' * 64
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'preprocessing hash'):
            self.bundle()
        self.inventory['preprocessing_sha256'] = canonical_hash(self.metadata['preprocessing'])
        self.inventory['files'].append(self.inventory['files'][0])
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Duplicate cache inventory'):
            self.bundle()

    def test_rejects_resplitting_subsampling_and_wrong_camera(self):
        for name, value in (('split_file', 'new.json'), ('frame_stride', 2),
                            ('max_frames_per_episode', 1), ('camera', 'hand_camera')):
            config = deepcopy(self.config)
            config['data'][name] = value
            with self.subTest(setting=name), self.assertRaises(ValueError):
                self.bundle(config)

    def test_dataset_identity_is_independent_of_machine_path(self):
        import shutil
        expected = self.bundle().identity['dataset_sha256']
        with tempfile.TemporaryDirectory() as directory:
            new_root = Path(directory) / 'copied'
            shutil.copytree(self.root, new_root)
            config = deepcopy(self.config)
            config['data']['cache_dir'] = str(new_root)
            self.assertEqual(self.bundle(config).identity['dataset_sha256'], expected)

    def test_changed_graph_settings_are_not_silently_ignored(self):
        config = deepcopy(self.config)
        config['graph']['intra_k'] = 2
        with self.assertRaisesRegex(ValueError, 'graph settings differ'):
            self.bundle(config)

    def test_main_config_dispatches_the_regular_trainer(self):
        with patch('training.gnn.trainer.train') as train:
            run('graph/configs/train_gated_mpnn_cached.yaml')
        self.assertEqual(train.call_args.args[0]['data']['factory'],
                         'training.gnn.cached_data:build_data')


if __name__ == '__main__':
    unittest.main()
