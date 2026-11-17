"""Family annotations must preserve geometry and reject unsupported GT proximity."""

from dataclasses import fields
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from data_collection.vertex_family import select_development_frames
from data_collection.vertex_family_labels import (
    UNKNOWN, EXTERIOR, RIM, CertificationSettings, detector_families,
    oracle_families, physical_references, project_world,
)
from evaluation.gnn.box_hole_diagnostics import graph_from_arrays
from graph.vertex_family import with_family_tokens
from training.gnn.data import TrainingBatch, canonical_hash, save_cached_sample
from training.gnn.vertex_family import initialization_hash
from training.gnn.vertex_family_data import VertexFamilyDataset
from data_collection.vertex_family_manifest import read_manifest
from evaluation.gnn.vertex_family import semantic_intervention
from data_collection.episode_dataset import file_sha256


def synthetic_front_face():
    """Render an analytic front face with a square aperture into native raw RGB-D."""
    intrinsic = np.array([[500., 0, 256], [0, 500., 256], [0, 0, 1]])
    extrinsic = np.array([[0., 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, .6]])
    gt = dict(box_pose=np.array([[0., 0, 0, 1, 0, 0, 0]]),
              hole_pose=np.array([[0., .02, .01, 1, 0, 0, 0]]),
              peg_half_sizes=np.array([[.1, .02, .02]]), hole_radius=np.array([.025]))
    y, x = np.indices((512, 512))
    world_y = (x - 256) * .5 / 500
    world_z = (y - 256) * .5 / 500
    solid = ((np.abs(world_y) <= .1) & (np.abs(world_z) <= .1)
             & ~((np.abs(world_y - .02) < .025) & (np.abs(world_z - .01) < .025)))
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    rgb[solid] = [255, 210, 137]
    depth = np.where(solid, 500, 800).astype(np.int16)
    return dict(rgb=rgb, depth=depth, intrinsic_cv=intrinsic, extrinsic_cv=extrinsic), gt


class VertexFamilyTests(unittest.TestCase):
    def test_manifest_reader_preserves_hash_and_split_validation(self):
        from copy import deepcopy
        from data_collection.episode_dataset import canonical_hash as dataset_hash
        from training.gnn.vertex_family_data import read_manifest as legacy_reader

        self.assertIs(legacy_reader, read_manifest)
        self.assertIs(canonical_hash, dataset_hash)
        original = dict(original_pool='train_only', reserved_test_frames_read=0, samples=[
            dict(sample_id='train/frame', parent_id='train_parent', split='train'),
            dict(sample_id='validation/frame', parent_id='validation_parent', split='validation'),
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            valid = dict(original, dataset_sha256=dataset_hash(original))
            path.write_text(json.dumps(valid))
            self.assertEqual(read_manifest(directory), valid)
            tampered = dict(valid, reserved_test_frames_read=1)
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, 'dataset_sha256'):
                read_manifest(directory)
            for problem in ('pool', 'test_frames', 'duplicate', 'overlap', 'empty'):
                altered = deepcopy(original)
                if problem == 'pool':
                    altered['original_pool'] = 'test'
                elif problem == 'test_frames':
                    altered['reserved_test_frames_read'] = 1
                elif problem == 'duplicate':
                    altered['samples'].append(deepcopy(altered['samples'][0]))
                elif problem == 'overlap':
                    altered['samples'][1]['parent_id'] = 'train_parent'
                else:
                    altered['samples'].pop()
                altered['dataset_sha256'] = dataset_hash(altered)
                path.write_text(json.dumps(altered))
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    read_manifest(directory)

    def test_detector_labels_use_metadata_only(self):
        labels = detector_families([0, 1, 1, 1], [{}, {"contour_hole": False}, {"contour_hole": True}, {}])
        np.testing.assert_array_equal(labels, [UNKNOWN, EXTERIOR, RIM, UNKNOWN])

    def test_oracle_certifies_front_topology_and_rejects_occluded_back(self):
        observation, gt = synthetic_front_face()
        references = physical_references(gt)
        xyz = np.asarray([row['world'] for row in references])
        uv, _ = project_world(xyz, observation['intrinsic_cv'], observation['extrinsic_cv'])
        family, physical_id, _ = oracle_families(xyz, uv, np.ones(16), observation, gt)
        np.testing.assert_array_equal(family[:4], np.full(4, EXTERIOR))
        np.testing.assert_array_equal(family[8:12], np.full(4, RIM))
        np.testing.assert_array_equal(family[4:8], np.full(4, UNKNOWN))
        np.testing.assert_array_equal(family[12:], np.full(4, UNKNOWN))
        self.assertEqual((physical_id >= 0).sum(), 8)

    def test_exact_gt_proximity_without_raw_hole_topology_is_unknown(self):
        observation, gt = synthetic_front_face()
        reference = physical_references(gt)[8]
        xyz = reference['world'][None]
        uv, _ = project_world(xyz, observation['intrinsic_cv'], observation['extrinsic_cv'])
        observation['rgb'][:] = [255, 210, 137]
        observation['depth'][:] = 500
        family, _, reason = oracle_families(xyz, uv, [1], observation, gt)
        self.assertEqual(family[0], UNKNOWN)
        self.assertEqual(reason[0], 'raw_face_topology_unverified')

    def test_uncertain_coordinate_or_projection_is_not_forced_to_nearest(self):
        observation, gt = synthetic_front_face()
        reference = physical_references(gt)[8]
        xyz = reference['world'][None].copy()
        uv, _ = project_world(xyz, observation['intrinsic_cv'], observation['extrinsic_cv'])
        shifted = xyz.copy()
        shifted[0, 1] += .01
        family, _, _ = oracle_families(shifted, uv, [1], observation, gt)
        self.assertEqual(family[0], UNKNOWN)
        family, _, _ = oracle_families(xyz, uv + 10, [1], observation, gt)
        self.assertEqual(family[0], UNKNOWN)

    def test_semantic_views_change_no_graph_geometry_or_padding(self):
        graph = graph_from_arrays(np.array([[[0., 0, 0], [1, 0, 0], [0, 0, 0]]]),
                                  np.array([[0, 1, -1]]), np.zeros((1, 3)))
        for mode in ('baseline', 'detector', 'oracle', 'type_control'):
            changed = with_family_tokens(graph, np.array([-1, 1, -1]), mode)
            changed.validate(num_semantic_ids=16)
            for field in fields(graph):
                if field.name not in ('semantic_id', 'semantic_valid'):
                    self.assertTrue(torch.equal(getattr(graph, field.name), getattr(changed, field.name)))
            self.assertEqual(changed.semantic_id[0, 2], -1)
            if mode in ('detector', 'oracle'):
                self.assertEqual(changed.semantic_id[0, 0], -1)
                self.assertEqual(changed.semantic_id[0, 1], 1)
        self.assertFalse(graph.semantic_valid.any())

    def test_parent_sampler_never_selects_original_validation_or_test(self):
        splits = dict(train=['train_a', 'train_b', 'train_c'], validation=['val'], test=['test'])
        episodes = sum(splits.values(), [])
        identity = dict(splits=splits, frames=[dict(id=f'{episode}/frame_{index:06d}.npz', sha256='x')
                                             for episode in episodes for index in range(5)])
        parents = {'parent_by_episode': {episode: episode for episode in episodes}}
        config = {'selection': dict(seed=91374, train_parents=2, validation_parents=1, frames_per_parent=3)}
        rows, new_splits = select_development_frames(config, identity, parents)
        self.assertEqual(len(rows), 9)
        self.assertEqual(set(row['episode'] for row in rows), set(splits['train']))
        self.assertFalse(set(new_splits['train']) & set(new_splits['validation']))
        parents['parent_by_episode']['val'] = 'train_a'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            select_development_frames(config, identity, parents)

    def test_all_frames_preserve_existing_validation_selection(self):
        episodes = ['train_a', 'train_b', 'train_c']
        identity = dict(splits=dict(train=episodes, validation=['val'], test=['test']),
                        frames=[dict(id=f'{episode}/frame_{index:06d}.npz', sha256='x')
                                for episode in episodes for index in range(5)])
        parents = {'parent_by_episode': {episode: episode for episode in episodes + ['val', 'test']}}
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / 'selection.json'
            previous.write_text(json.dumps({'splits': {'validation': ['train_b']}}))
            config = {'selection': dict(seed=91374, train_parents=2, validation_parents=1,
                                        frames_per_parent=None, validation_selection=str(previous))}
            rows, splits = select_development_frames(config, identity, parents)
            self.assertEqual(len(rows), 15)
            self.assertEqual(splits['validation'], ['train_b'])
            self.assertEqual(set(splits['train']), {'train_a', 'train_c'})
            self.assertEqual(sum(row['split'] == 'validation' for row in rows), 5)

    def test_graph_cache_reuses_only_exact_detector_inputs(self):
        from types import SimpleNamespace
        from data_collection.vertex_family_cache import ExistingGraphCache
        from graph.build_graph import GraphBuilder
        builder = GraphBuilder()
        result = SimpleNamespace(xyz=torch.tensor([[0., 0., 0.], [1., 0., 0.]]),
                                 object_id=torch.tensor([0, 1]), metadata=[{}, {}])
        tcp = torch.zeros(1, 3)
        target = torch.zeros(1, 21)
        sample = TrainingBatch(builder([result], tcp), {'object21': target}, ['sample'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.npz'
            identity = dict(frame_sha256='source', sample_id='sample', preprocessing_sha256='preprocess')
            save_cached_sample(path, sample, identity)
            cache = ExistingGraphCache.__new__(ExistingGraphCache)
            cache.builder = builder
            cache.root = Path(directory)
            cache.preprocessing = 'preprocess'
            cache.rows = {'sample': dict(file='graph.npz', sha256=file_sha256(path))}
            row = dict(sample_id='sample', source_sha256='source')
            self.assertIsNotNone(cache.find(row, result, tcp, target))
            result.xyz[0, 0] = .01
            self.assertIsNone(cache.find(row, result, tcp, target))
            result.xyz[0, 0] = 0
            self.assertIsNone(cache.find(row, result, tcp, target + 1))

    def test_nonpositive_certification_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            CertificationSettings(coordinate_tolerance_m=0)

    def test_switching_family_features_preserves_fresh_initialization(self):
        model = dict(name='vertex_edge_gated_mpnn', num_semantic_ids=16, hidden=16,
                     layers=1, edge_embedding_dim=4, use_semantic_ids=False)
        baseline_hash = initialization_hash(model, 0)
        model['use_semantic_ids'] = True
        self.assertEqual(baseline_hash, initialization_hash(model, 0))
        self.assertNotEqual(baseline_hash, initialization_hash(model, 1))

    def test_label_intervention_preserves_unknown_and_coordinates(self):
        graph = graph_from_arrays(np.array([[[0., 0, 0], [1, 0, 0], [2, 0, 0]]]),
                                  np.array([[0, 1, 1]]), np.zeros((1, 3)))
        annotated = with_family_tokens(graph, [-1, 0, 1], 'oracle')
        swapped = semantic_intervention(annotated, 'swap')
        np.testing.assert_array_equal(swapped.semantic_id, [[-1, 1, 0]])
        self.assertTrue(torch.equal(swapped.xyz_world, annotated.xyz_world))
        self.assertTrue(torch.equal(swapped.semantic_valid, annotated.semantic_valid))
        self.assertFalse(semantic_intervention(annotated, 'erase').semantic_valid.any())

    def test_dataset_verifies_hashes_and_keeps_oracle_labels_out_of_student_view(self):
        graph = graph_from_arrays(np.array([[[0., 0, 0], [.1, 0, 0]]]),
                                  np.array([[0, 1]]), np.zeros((1, 3)))
        target = torch.zeros(1, 21)
        sample = TrainingBatch(graph, {'object21': target}, ['ep/frame.npz'])
        identity = {'sample_id': 'ep/frame.npz'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_cached_sample(root / 'graph.npz', sample, identity)
            np.savez_compressed(root / 'labels.npz', detector_family=[-1, 0], oracle_family=[-1, 1],
                                xyz_world=graph.xyz_world[0].numpy())
            row = dict(sample_id='ep/frame.npz', parent_id='p1', split='train', graph_path='graph.npz',
                       labels_path='labels.npz', sample_identity=identity,
                       graph_sha256=file_sha256(root / 'graph.npz'), labels_sha256=file_sha256(root / 'labels.npz'))
            manifest = dict(samples=[row])
            detector = VertexFamilyDataset(root, manifest, 'train', 'detector')
            oracle = VertexFamilyDataset(root, manifest, 'train', 'oracle')
            self.assertEqual(detector[0].graph.semantic_id[0, 1], 0)
            self.assertEqual(oracle[0].graph.semantic_id[0, 1], 1)
            self.assertTrue(torch.equal(detector[0].graph.xyz_world, oracle[0].graph.xyz_world))
            (root / 'labels.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                VertexFamilyDataset(root, manifest, 'train', 'detector')


if __name__ == '__main__':
    unittest.main()
