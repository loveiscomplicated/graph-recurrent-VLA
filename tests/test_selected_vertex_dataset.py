"""Physical selection must keep only consistently approved, unique samples."""

import json
from pathlib import Path
import tempfile
import unittest

from data_collection.selected_vertex_dataset import selected_frames


class SelectedVertexDatasetTests(unittest.TestCase):
    def test_selected_frames_reject_duplicates_and_inconsistent_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'frames.jsonl'
            good = dict(sample_id='a', peg_pass=True, hole_pass=True, both_pass=True)
            rejected = dict(sample_id='b', peg_pass=True, hole_pass=False, both_pass=False)
            path.write_text('\n'.join(json.dumps(row) for row in (good, rejected)))
            self.assertEqual(selected_frames(path), [good])
            path.write_text('\n'.join(json.dumps(row) for row in (good, good)))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                selected_frames(path)
            path.write_text(json.dumps(dict(good, both_pass=False)))
            with self.assertRaisesRegex(ValueError, 'Inconsistent'):
                selected_frames(path)

class SelectedDatasetIntegrationTests(unittest.TestCase):
    def test_prepares_train_and_validation_graphs_for_existing_loader(self):
        from unittest.mock import patch
        import numpy as np
        import torch
        from data_collection.episode_dataset import file_sha256
        from data_collection.selected_vertex_dataset import prepare
        from evaluation.gnn.box_hole_diagnostics import graph_from_arrays
        from training.gnn.data import TrainingBatch, canonical_hash, save_cached_sample
        from training.gnn.vertex_family_data import VertexFamilyDataset, build_data
        from data_collection.vertex_family_manifest import read_manifest

        graph = graph_from_arrays(np.array([[[0., 0, 0], [1, 0, 0]]]),
                                  np.array([[0, 1]]), np.zeros((1, 3)))
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ', {'WANDB_MODE': 'disabled'}):
            root = Path(temp)
            cache = root / 'cache'
            (cache / 'metadata').mkdir(parents=True)
            identity = {'sample_id': 'train/frame', 'frame_sha256': 'a' * 64}
            source = cache / 'graph.npz'
            save_cached_sample(source, TrainingBatch(graph, {'object21': torch.zeros(1, 21)}, ['train/frame']), identity)
            (cache / 'metadata/cache_inventory.json').write_text(json.dumps({'files': [
                {'sample_id': 'train/frame', 'file': 'graph.npz', 'sha256': file_sha256(source)}]}))
            selection = root / 'selection.jsonl'
            selection.write_text(json.dumps(dict(sample_id='train/frame', parent_id='p1', sha256='a' * 64,
                                                 peg_pass=True, hole_pass=True, both_pass=True)))
            validation = root / 'validation'
            validation.mkdir()
            val_identity = {'sample_id': 'validation/frame'}
            val_graph = validation / 'graph.npz'
            save_cached_sample(val_graph, TrainingBatch(graph, {'object21': torch.zeros(1, 21)},
                                                         ['validation/frame']), val_identity)
            labels = validation / 'labels.npz'
            np.savez_compressed(labels, xyz_world=graph.xyz_world[0].numpy(), detector_family=[-1, -1])
            val_row = dict(sample_id='validation/frame', parent_id='p2', split='validation',
                           graph_path='graph.npz', graph_sha256=file_sha256(val_graph),
                           labels_path='labels.npz', labels_sha256=file_sha256(labels), sample_identity=val_identity)
            dummy = dict(val_row, sample_id='unused/frame', parent_id='p3', split='train')
            manifest = dict(original_pool='train_only', reserved_test_frames_read=0,
                            samples=[dummy, val_row], source_dataset_sha256='source')
            manifest['dataset_sha256'] = canonical_hash(manifest)
            (validation / 'manifest.json').write_text(json.dumps(manifest))
            output = prepare(dict(cache_root=str(cache), selection_frames=str(selection),
                                  validation_dataset=str(validation), output={'root': str(root / 'output')}))
            result = read_manifest(output)
            train = VertexFamilyDataset(output, result, 'train', 'baseline')
            valid = VertexFamilyDataset(output, result, 'validation', 'baseline')
            self.assertEqual((len(train), len(valid)), (1, 1))
            self.assertTrue(torch.equal(train[0].graph.xyz_world, graph.xyz_world))
            self.assertFalse(train[0].graph.semantic_valid.any())
            validation_selection = root / 'validation_selection.jsonl'
            validation_selection.write_text(json.dumps(dict(sample_id='validation/frame',
                peg_pass=True, hole_pass=True, both_pass=True)))
            config = {'data': {'family_dataset': str(output), 'family_mode': 'baseline'},
                      'training': {'batch_size': 1}}
            original = build_data(config, torch.Generator())
            config['data']['validation_selection'] = str(validation_selection)
            selected = build_data(config, torch.Generator())
            self.assertEqual(selected.identity['sample_counts'], {'train': 1, 'validation': 1})
            self.assertEqual(selected.identity['resume_from_dataset_sha256'], original.identity['dataset_sha256'])
            self.assertNotEqual(selected.identity['dataset_sha256'], original.identity['dataset_sha256'])
            validation_selection.write_text(json.dumps(dict(sample_id='unknown',
                peg_pass=True, hole_pass=True, both_pass=True)))
            with self.assertRaisesRegex(ValueError, 'every validation sample'):
                build_data(config, torch.Generator())
