"""Behavior contracts for raw, cached and episode physical-vertex selection."""

from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data_collection import physical_vertex_selection as raw
from data_collection import physical_vertex_cache_selection as cached
from data_collection import physical_vertex_episode_selection as episodes
from data_collection.episode_dataset import file_sha256
from data_collection.vertex_family_labels import SIGNS, project_world


def decision(row):
    """Generate ordered, mixed decisions without invoking a detector."""
    passed = int(Path(row['sample_id']).stem) % 2 == 0
    return dict(sample_id=row['sample_id'], parent_id=row['parent_id'], sha256=row['sha256'],
                peg_pass=passed, hole_pass=True, both_pass=passed, peg_single_face=False,
                unique_peg_ids=[0, 1, 2, 4] if passed else [0], unique_rim_ids=[8, 9, 10])


class InlinePool:
    """Keep orchestration deterministic; actual spawn workers are tested separately."""

    def __init__(self, **kwargs):
        self.options = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, rows, chunksize):
        return map(decision, rows)


class LocalLogger:
    def __init__(self, **kwargs):
        self.run_id = 'test_run'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def link_output(self, *args):
        pass

    def record(self, **kwargs):
        pass


class PhysicalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_json(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return str(path)

    def selection_config(self, suffix, max_frames=None):
        names = [f'episode_{i}' for i in range(630)]
        identity = dict(splits={'train': names}, frames=[
            dict(id=f'{name}/{i}.npz', sha256=str(i)) for i, name in enumerate(names)])
        config = dict(source_identity=self.write_json('identity.json', identity),
                      source_parent_manifest=self.write_json('parents.json', {'parent_by_episode': dict(zip(names, names))}),
                      development_selection=self.write_json('selection.json', {'splits': {'validation': names[:40]}}),
                      max_frames=max_frames, workers=2, output={'root': str(self.root / suffix)},
                      cache_root=str(self.root / 'cache'))
        inventory = {'files': [dict(sample_id=row['id'], file=f'{i}.npz', sha256=str(i))
                               for i, row in enumerate(identity['frames'])]}
        self.write_json('cache/metadata/cache_inventory.json', inventory)
        return config

    def run_inline(self, module, config):
        with ExitStack() as stack:
            # The cached compatibility module may delegate orchestration to raw.
            for target in (raw, cached, episodes):
                stack.enter_context(patch.object(target, 'ExperimentLogger', LocalLogger, create=True))
                stack.enter_context(patch.object(target, 'ProcessPoolExecutor', InlinePool, create=True))
            return module.prepare(config)

    def test_raw_and_cached_population_order_counts_and_artifacts(self):
        for module in (raw, cached):
            for limit in (None, 3):
                with self.subTest(module=module.__name__, limit=limit):
                    config = self.selection_config(f'{module.__name__}_{limit}', limit)
                    output = self.run_inline(module, config)
                    rows = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines()]
                    expected_ids = [f'episode_{i}/{i}.npz' for i in range(40, 630)][:limit]
                    self.assertEqual([r['sample_id'] for r in rows], expected_ids)
                    self.assertEqual(rows, [decision(dict(sample_id=x, parent_id=str(Path(x).parent),
                                                         sha256=Path(x).stem)) for x in expected_ids])
                    summary = json.loads((output / 'summary.json').read_text())
                    self.assertEqual(summary['frames'], len(expected_ids))
                    self.assertEqual(summary['parents'], len(expected_ids))
                    self.assertEqual(summary['both_pass'], 295 if limit is None else 2)
                    self.assertEqual(summary['complete'], limit is None)
                    self.assertEqual(json.loads((output / 'effective_config.json').read_text()), config)
                    provenance = json.loads((output / 'provenance.json').read_text())
                    self.assertEqual('cache_inventory' in provenance, module is cached)
                    self.assertEqual('pixel_source' in provenance, module is cached)

    def test_invalid_inputs_and_inventory_fail_before_selection(self):
        for module in (raw, cached):
            config = self.selection_config(module.__name__)
            for key in ('workers', 'max_frames'):
                with self.subTest(module=module.__name__, key=key):
                    with self.assertRaisesRegex(ValueError, key):
                        self.run_inline(module, dict(config, **{key: 0}))
        config = self.selection_config('invalid_cache')
        inventory_path = self.root / 'cache/metadata/cache_inventory.json'
        inventory = json.loads(inventory_path.read_text())
        inventory['files'].append(inventory['files'][0])
        inventory_path.write_text(json.dumps(inventory))
        with self.assertRaisesRegex(ValueError, 'Duplicate sample IDs'):
            self.run_inline(cached, config)

    def test_episode_original_decisions_and_new_frame_are_distinct(self):
        ids = [f'episode/{i}.npz' for i in range(961)]
        prediction = self.root / 'predictions.npz'
        np.savez(prediction, sample_ids=ids)
        self.write_json('cache/metadata/data_identity.json', {'frames': [dict(id=x, sha256='raw') for x in ids]})
        self.write_json('cache/metadata/cache_inventory.json', {'files': [dict(sample_id=x, file='cache.npz', sha256='cache') for x in ids]})
        previous = [decision(dict(sample_id=x, parent_id='episode', sha256='raw')) for x in ids[:960]]
        previous[0].update(peg_pass=False, both_pass=False, unique_peg_ids=[0])
        path = self.root / 'previous.jsonl'
        path.write_text('\n'.join(json.dumps(row) for row in previous))
        config = dict(predictions=str(prediction), source_root=str(self.root), cache_root=str(self.root / 'cache'),
                      parent_manifest=self.write_json('parents.json', {'parent_by_episode': {'episode': 'episode'}}),
                      previous_selection=str(path), output={'root': str(self.root / 'episode_output')})
        output = self.run_inline(episodes, config)
        rows = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0], dict(previous[0], decision_source='original_detector_uv'))
        self.assertEqual(rows[-1]['decision_source'], 'projected_cached_xyz')
        self.assertTrue(rows[-1]['both_pass'])
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual((summary['frames'], summary['original_frames'], summary['mismatches'], summary['both_pass']), (961, 960, 1, 480))
        mismatches = json.loads((output / 'mismatches.json').read_text())
        self.assertEqual(mismatches[0]['fields'], ['peg_pass', 'both_pass'])
        path.write_text(json.dumps(previous[0]))
        with self.assertRaisesRegex(ValueError, '960 validation decisions'):
            self.run_inline(episodes, config)

    def test_parent_partition_and_unsafe_source_or_cache_paths_are_rejected(self):
        config = self.selection_config('bad_paths', max_frames=1)
        identity = json.loads(Path(config['source_identity']).read_text())
        parents = json.loads(Path(config['source_parent_manifest']).read_text())
        selection = json.loads(Path(config['development_selection']).read_text())
        identity['frames'].append(dict(identity['frames'][0]))
        with self.assertRaisesRegex(ValueError, 'Invalid or duplicate'):
            raw.select_source_rows(identity, parents, selection)
        identity['frames'].pop()
        identity['frames'][-1]['id'] = '../outside.npz'
        with self.assertRaisesRegex(ValueError, 'Invalid or duplicate'):
            raw.select_source_rows(identity, parents, selection)
        selection['splits']['validation'].pop()
        with self.assertRaisesRegex(ValueError, '40 development'):
            raw.select_source_rows(identity, parents, selection)
        inventory_path = self.root / 'cache/metadata/cache_inventory.json'
        inventory = json.loads(inventory_path.read_text())
        # Inventory validation must still precede the max_frames truncation.
        inventory['files'][-1]['file'] = '../outside.npz'
        inventory_path.write_text(json.dumps(inventory))
        with self.assertRaisesRegex(ValueError, 'relative to cache_root'):
            self.run_inline(cached, config)

    def test_spawn_workers_use_native_frames_and_preserve_order(self):
        """Exercise real detector/certification and pickling across two CPU workers."""
        pose = np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.float32)
        frame = self.root / 'frame.npz'
        image_prefix = 'observation/sensor_data/base_camera/'
        camera_prefix = 'observation/sensor_param/base_camera/'
        np.savez_compressed(frame, **{
            image_prefix + 'rgb': np.zeros((1, 512, 512, 3), np.uint8),
            image_prefix + 'depth': np.zeros((1, 512, 512, 1), np.int16),
            camera_prefix + 'intrinsic_cv': np.array([[[100, 0, 256], [0, 100, 256], [0, 0, 1]]]),
            camera_prefix + 'extrinsic_cv': np.eye(4)[None, :3],
            'ground_truth/peg_pose': pose[None],
            'ground_truth/box_pose': pose[None],
            'ground_truth/hole_pose': pose[None],
            'ground_truth/peg_half_sizes': np.array([[.1, .02, .02]]),
            'ground_truth/hole_radius': np.array([.03]),
        })
        archive = self.root / 'graph.npz'
        points = SIGNS[[0, 1, 2, 4]] * [.1, .02, .02] + pose[:3]
        np.savez(archive, node_valid=[[True] * 4], xyz_world=[points], object_type=[[0] * 4])
        config = dict(source_root=str(self.root), cache_root=str(self.root), camera='base_camera',
                      workers=2, detector={'device': 'cpu'})
        for initializer, inspector, expect_peg in (
            (raw.initialize_worker, raw.inspect_frame, False),
            (cached.initialize_worker, cached.inspect_frame, True),
        ):
            rows = [dict(sample_id=frame.name, parent_id=f'parent_{i}', sha256=file_sha256(frame))
                    for i in range(10)]
            if expect_peg:
                for row in rows:
                    row.update(cache_file=archive.name, cache_sha256=file_sha256(archive))
            results = list(raw.iter_frame_decisions(rows, config, initializer, inspector))
            self.assertEqual([r['parent_id'] for r in results], [r['parent_id'] for r in rows])
            self.assertEqual([r['peg_pass'] for r in results], [expect_peg] * 10)
            self.assertFalse(any(r['hole_pass'] for r in results))
            self.assertFalse(any(r['both_pass'] for r in results))

    def test_worker_pixel_sources_padding_and_checksum_rejection(self):
        gt = dict(peg_pose=np.array([0, 0, 1, 1, 0, 0, 0]), peg_half_sizes=np.array([.1, .02, .02]))
        observation = dict(rgb=np.zeros((512, 512, 3), np.uint8), depth=np.zeros((512, 512, 1), np.int16),
                           intrinsic_cv=np.array([[100, 0, 256], [0, 100, 256], [0, 0, 1]]), extrinsic_cv=np.eye(4)[:3])
        xyz = SIGNS[[0, 1, 2, 4]] * gt['peg_half_sizes'] + gt['peg_pose'][:3]
        uv, _ = project_world(xyz, observation['intrinsic_cv'], observation['extrinsic_cv'])
        wrong_uv = uv + 10
        frame = self.root / 'frame.npz'
        frame.write_bytes(b'raw fixture')
        archive = self.root / 'graph.npz'
        np.savez(archive, node_valid=[[True, True, True, True, False]],
                 xyz_world=[np.vstack((xyz, [999, 999, 999]))], object_type=[[0, 0, 0, 0, 1]])
        row = dict(sample_id=frame.name, parent_id='parent', sha256=file_sha256(frame),
                   cache_file=archive.name, cache_sha256=file_sha256(archive))
        from types import SimpleNamespace
        detected = SimpleNamespace(xyz=torch.tensor(xyz), uv=torch.tensor(wrong_uv), object_id=torch.zeros(4))
        state = dict(root=self.root, cache_root=self.root, camera='base_camera',
                     certification=raw.CertificationSettings(), detector=lambda *args: detected)
        with ExitStack() as stack:
            worker_modules = {importlib.import_module(worker.__module__)
                              for worker in (raw.inspect_frame, cached.inspect_frame)}
            for module in worker_modules:
                stack.enter_context(patch.dict(module._WORKER, state, clear=True))
                stack.enter_context(patch.object(module, 'read_frame', return_value=(observation, gt), create=True))
                stack.enter_context(patch.object(module, 'oracle_families', return_value=(np.ones(4), np.array([8, 9, 10, 10]), None), create=True))
            original = raw.inspect_frame(dict(sample_id=frame.name, parent_id='parent', sha256=row['sha256']))
            projected = cached.inspect_frame(row)
            self.assertFalse(original['peg_pass'])
            self.assertTrue(projected['both_pass'])
            self.assertNotIn('cache_file', projected)
            with self.assertRaisesRegex(ValueError, 'Cached graph missing or hash mismatch'):
                cached.inspect_frame(dict(row, cache_sha256='wrong'))
            with self.assertRaisesRegex(ValueError, 'Raw frame missing or hash mismatch'):
                raw.inspect_frame(dict(row, sha256='wrong'))


if __name__ == '__main__':
    unittest.main()
