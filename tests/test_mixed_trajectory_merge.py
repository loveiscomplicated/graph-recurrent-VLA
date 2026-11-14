"""Check partition coverage and preservation of interrupted pilot artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_collection.episode_dataset import file_sha256
from data_collection.mixed_trajectory_merge import merge


class MergeTest(unittest.TestCase):
    def test_merge_preserves_overlaps_and_worker_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, worker, overlap = base / 'pilot', base / 'worker', base / 'overlap'
            root.mkdir()
            worker.mkdir()
            for name in ('collector_source.py', 'worker_collector_source.py'):
                (root / name).write_text('# saved collector source\n')
            protocol = {'generator': 1, 'records': [{'seed': 11, 'split': 'pilot'},
                                                      {'seed': 22, 'split': 'pilot'}],
                        'modes': ['expert', 'random', 'expert_random']}
            for directory in (root, worker):
                (directory / 'protocol.json').write_text(json.dumps(protocol))
            digest = file_sha256(root / 'protocol.json')
            old, new = [], []
            for seed, directory, rows in ((11, root, old), (22, worker, new)):
                for mode in protocol['modes']:
                    name = f'pilot_{seed}_{mode}'
                    path = directory / name
                    path.mkdir()
                    (path / 'frame_000000.npz').write_bytes(f'{seed}-{mode}'.encode())
                    rows.append({'seed': seed, 'mode': mode, 'path': name})
            extra = root / 'pilot_22_expert'
            extra.mkdir()
            (extra / 'old_frame').write_text('preserved')
            old.append({'seed': 22, 'mode': 'expert', 'path': extra.name})
            partial = root / 'pilot_22_random.partial'
            partial.mkdir()
            (partial / 'unfinished').write_text('preserved')
            (root / 'manifest.json').write_text(json.dumps({'status': 'ERROR', 'protocol_sha256': digest,
                                                          'episodes': old}))
            (worker / 'manifest.json').write_text(json.dumps({'status': 'COMPLETE', 'episodes': new}))
            config = {'collection_root': str(root), 'sequential_seed_count': 1,
                      'worker_roots': [str(worker)], 'overlap_archive': str(overlap)}
            logger = MagicMock()
            with patch('data_collection.mixed_trajectory_merge.ExperimentLogger') as factory:
                factory.return_value.__enter__.return_value = logger
                merge(config)
            result = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(len(result['episodes']), 6)
            self.assertEqual((overlap / extra.name / 'old_frame').read_text(), 'preserved')
            self.assertEqual((overlap / partial.name / 'unfinished').read_text(), 'preserved')
            for row in new:
                original = worker / row['path'] / 'frame_000000.npz'
                merged = root / row['path'] / 'frame_000000.npz'
                self.assertEqual(original.read_bytes(), merged.read_bytes())
                self.assertEqual(original.stat().st_ino, merged.stat().st_ino)


if __name__ == '__main__':
    unittest.main()
