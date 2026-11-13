"""Parallel/serial parity and interrupted source preservation contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.temporal.corner_tracks import load_dataset, prepare as serial_prepare
from data_collection.temporal.corner_tracks_parallel import prepare as parallel_prepare
from tests.fixtures.corner_tracking import write_corner_family_source


class ParallelPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'family'
        self.manifest = write_corner_family_source(self.source)
        self.config = {
            'source': {'family_dataset': str(self.source)},
            'output': {'root': str(self.root / 'prepared')},
            'seed': 0,
            'workers': 2,
        }
        environment = patch.dict('os.environ', {
            'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(self.root / 'experiments'),
        })
        environment.start()
        self.addCleanup(environment.stop)

    def source_hashes(self):
        return {path: file_sha256(path) for path in self.source.rglob('*.npz')}

    def test_serial_and_parallel_preserve_identical_frames_and_source_bytes(self):
        original_hashes = self.source_hashes()
        serial, serial_manifest = load_dataset(serial_prepare(self.config))
        parallel, parallel_manifest = load_dataset(parallel_prepare(self.config))
        self.assertEqual(serial_manifest['provenance']['source_dataset_sha256'],
                         parallel_manifest['provenance']['source_dataset_sha256'])
        for split in ('train', 'calibration', 'validation'):
            self.assertEqual(len(serial[split]), len(parallel[split]))
            for serial_episode, parallel_episode in zip(serial[split], parallel[split]):
                self.assertEqual((serial_episode.name, serial_episode.parent, serial_episode.split),
                                 (parallel_episode.name, parallel_episode.parent, parallel_episode.split))
                self.assertEqual(len(serial_episode.frames), len(parallel_episode.frames))
                for left, right in zip(serial_episode.frames, parallel_episode.frames):
                    self.assertEqual(left.sample_id, right.sample_id)
                    for name in ('xyz', 'kind', 'confidence', 'confidence_valid', 'robot'):
                        np.testing.assert_array_equal(getattr(left.observation, name),
                                                      getattr(right.observation, name))
                    for name in ('physical_ids', 'reference_xyz', 'reference_kind', 'target_object21'):
                        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
        self.assertEqual(self.source_hashes(), original_hashes)

    def test_corrupt_source_leaves_no_completion_manifest(self):
        original_hashes = self.source_hashes()
        raw_path = next((self.source / 'raw').rglob('*.npz'))
        original = raw_path.read_bytes()
        try:
            raw_path.write_bytes(original + b'corrupted')
            with self.assertRaisesRegex(ValueError, 'source path/hash mismatch'):
                parallel_prepare(self.config)
        finally:
            raw_path.write_bytes(original)

        outputs = list((self.root / 'prepared').iterdir())
        self.assertEqual(len(outputs), 1)
        self.assertFalse((outputs[0] / 'manifest.json').exists())
        self.assertEqual(self.source_hashes(), original_hashes)

    def test_both_adapters_reject_noncontiguous_episodes(self):
        missing = self.manifest['samples'][1]['sample_id']
        self.manifest['samples'] = [row for row in self.manifest['samples'] if row['sample_id'] != missing]
        self.manifest.pop('dataset_sha256')
        self.manifest['dataset_sha256'] = canonical_hash(self.manifest)
        (self.source / 'manifest.json').write_text(json.dumps(self.manifest))
        for prepare in (serial_prepare, parallel_prepare):
            with self.subTest(adapter=prepare.__module__):
                with self.assertRaisesRegex(ValueError, 'contiguous full episodes'):
                    prepare(self.config)
        self.assertFalse((self.root / 'prepared').exists())


if __name__ == '__main__':
    unittest.main()
