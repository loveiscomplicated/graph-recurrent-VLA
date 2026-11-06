"""Training contracts; opt-in optimizer tests are reserved for the training host."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from graph.build_graph import GraphBuilder
from graph.utils.object21_targets import object21_from_gt
from main import load_config
from training.gnn.checkpoint import load_checkpoint, save_checkpoint
from training.gnn.data import (
    CachedEpisodeGraphDataset, DataBundle, TrainingBatch, collate_graph_samples,
    split_episodes,
)
from training.gnn.tasks import Object21Task, StepResult
from training.gnn.trainer import build_model, resolve_config, run_epoch, train


def sample():
    observation = SimpleNamespace(
        xyz=torch.tensor([[0., 0, 0], [0.04, 0, 0], [0.1, 0, 0]]),
        object_id=torch.tensor([0, 0, 1]), metadata=[{}, {}, {}],
    )
    graph = GraphBuilder()([observation], torch.zeros(1, 3))
    target = torch.zeros(1, 21)
    target[:, 3] = target[:, 13] = 1
    target[:, 17:21] = torch.tensor([0.04, 0.01, 0.01, 0.02])
    return TrainingBatch(graph, {'object21': target}, ['fixture/frame_000000.npz'])


def fixture_data(config, generator):
    """Synthetic data factory to exercise the loop independently of the detector."""
    return DataBundle([sample()], [sample()], {'dataset_sha256': 'synthetic-test-only'})


class NodeTask:
    """A tensor-output BasicGNN task, with a node/coordinate denominator."""
    def compute(self, model, batch):
        output = model(batch.graph)[batch.graph.node_valid]
        loss = output.square().mean()
        return StepResult(loss, {'loss': float(loss.detach())}, {'loss': output.numel()})


class TrainingContracts(unittest.TestCase):
    def test_targets_keep_world_axes_and_use_local_positive_x_tip(self):
        root_half = 2**-0.5
        gt = {'ee_pose': [[1, 2, 3, 0, 1, 0, 0]],
              'peg_pose': [[2, 3, 4, root_half, 0, 0, root_half]],
              'hole_pose': [[3, 5, 7, 1, 0, 0, 0]],
              'peg_half_sizes': [[0.04, 0.01, 0.01]], 'hole_radius': [0.02]}
        target = object21_from_gt(gt)
        torch.testing.assert_close(target[:, :3], torch.tensor([[1., 1, 1]]))
        torch.testing.assert_close(target[:, 7:10], torch.tensor([[-1., -1.96, -3]]))
        gt['peg_pose'][0][3:] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            object21_from_gt(gt)

    def test_split_is_disjoint_reproducible_and_rejects_insufficient_episodes(self):
        episodes = [SimpleNamespace(identity=f'episode_{i}') for i in range(10)]
        settings = {'validation_fraction': 0.2, 'test_fraction': 0.1}
        splits = split_episodes(episodes, settings, 4)
        self.assertEqual(splits, split_episodes(episodes, settings, 4))
        self.assertEqual(len(set(sum(splits.values(), []))), 10)
        with self.assertRaises(ValueError):
            split_episodes(episodes[:1], settings, 4)

    def test_collation_offsets_and_empty_filtering(self):
        first = sample()
        second = deepcopy(first)
        second.sample_ids = ['other']
        batch = collate_graph_samples([first, second])
        batch.graph.validate()
        self.assertEqual(batch.targets['object21'].shape, (2, 21))
        self.assertTrue((batch.graph.edge_index[:, 6:] >= 3).all())
        empty = deepcopy(first)
        empty.graph = GraphBuilder()([SimpleNamespace(xyz=torch.empty(0, 3),
            object_id=torch.empty(0, dtype=torch.long), metadata=[])], torch.zeros(1, 3))
        self.assertIsNone(collate_graph_samples([empty], empty_inputs='skip'))
        self.assertEqual(len(collate_graph_samples([first, empty]).sample_ids), 2)

    def test_cache_constructor_does_not_read_frames_or_write_files(self):
        config = resolve_config(load_config('graph/configs/train_gated_mpnn.yaml'))
        with tempfile.TemporaryDirectory() as directory:
            config['data']['cache_dir'] = str(Path(directory) / 'not_created')
            dataset = CachedEpisodeGraphDataset([], config, {'version': 1})
            self.assertEqual(len(dataset), 0)
            self.assertFalse(Path(config['data']['cache_dir']).exists())

    def test_cache_hit_and_miss_without_materializing_a_cache(self):
        config = resolve_config(load_config('graph/configs/train_gated_mpnn.yaml'))
        source = {'path': Path('/not-read/frame.npz'), 'id': 'episode/frame.npz', 'sha256': 'hash'}
        dataset = CachedEpisodeGraphDataset([source], config, {'version': 1})
        expected = sample()
        with patch('training.gnn.data.file_sha256', return_value='hash'), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('training.gnn.data.load_cached_sample', return_value=expected) as load, \
             patch('training.gnn.data.save_cached_sample') as save:
            self.assertIs(dataset[0], expected)
            load.assert_called_once()
            save.assert_not_called()
        with patch('training.gnn.data.file_sha256', return_value='hash'), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('training.gnn.data.EpisodeGraphDataset.__getitem__', return_value=expected), \
             patch('training.gnn.data.save_cached_sample') as save:
            self.assertIs(dataset[0], expected)
            save.assert_called_once()

    def test_both_output_types_use_the_same_validation_loop(self):
        config = resolve_config(load_config('graph/configs/train_gated_mpnn.yaml'))
        config['model'].update(hidden=16, layers=1)
        model = build_model(config['model'], torch.device('cpu'))
        result = run_epoch(model, [sample()], Object21Task(), torch.device('cpu'))
        self.assertIn('loss', result)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        basic = build_model({'name': 'basic_gnn', 'n_layers': 1, 'in_dim': 8,
                             'h_dim': 8, 'out_dim': 3, 'edge_dim': 8}, torch.device('cpu'))
        self.assertIn('loss', run_epoch(basic, [sample()], NodeTask(), torch.device('cpu')))

    def test_checkpoint_round_trip_rng_and_mismatch_rejection_without_training(self):
        config = resolve_config(load_config('graph/configs/train_gated_mpnn.yaml'))
        config['model'].update(hidden=16, layers=1)
        model = build_model(config['model'], torch.device('cpu'))
        optimizer = torch.optim.AdamW(model.parameters())
        generator = torch.Generator().manual_seed(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            save_checkpoint(path, model=model, optimizer=optimizer, scheduler=None, epoch=1,
                best_loss=1., best_model_state=model.state_dict(), config=config,
                dataset_sha256='fixture', device=torch.device('cpu'), shuffle_generator=generator)
            expected = torch.rand(4)
            arguments = dict(model=model, optimizer=optimizer, scheduler=None, config=config,
                dataset_sha256='fixture', device=torch.device('cpu'), shuffle_generator=generator)
            load_checkpoint(path, **arguments)
            torch.testing.assert_close(torch.rand(4), expected)
            arguments['dataset_sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_checkpoint(path, **arguments)

    def test_loss_settings_fail_before_data_execution(self):
        for settings in ({'pos_unit_m': 0}, {'consistency_weight': -1},
                         {'size_units_m': [1]}, {'pos_unit_m': float('nan')}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Object21Task(**settings)


@unittest.skipUnless(os.getenv('RUN_GNN_TRAINING_TESTS') == '1',
                     'Optimizer execution is opt-in on the training host')
class ServerTrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training_and_common_basic_task(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': directory + '/experiments'}):
            config = load_config('graph/configs/train_gated_mpnn.yaml')
            config['device'] = 'cpu'
            config['model'].update(hidden=16, layers=1, dropout=0.1)
            config['data'] = {'factory': 'tests.test_gnn_training:fixture_data'}
            config['output'] = {'root': directory + '/outputs'}
            config['training'] = {'epochs': 2, 'batch_size': 1}
            full = train(config)
            config['training']['epochs'] = 1
            first = train(config)
            config['training']['epochs'] = 2
            config['resume'] = str(first / 'last.pt')
            resumed = train(config)
            expected = torch.load(full / 'last.pt', weights_only=True)
            actual = torch.load(resumed / 'last.pt', weights_only=True)
            for name, weight in expected['model'].items():
                torch.testing.assert_close(actual['model'][name], weight, rtol=0, atol=0)
            config['resume'] = None
            config['model'] = {'name': 'basic_gnn', 'n_layers': 1, 'in_dim': 8,
                               'h_dim': 8, 'out_dim': 3, 'edge_dim': 8}
            config['task'] = {'factory': 'tests.test_gnn_training:NodeTask'}
            self.assertTrue((train(config) / 'last.pt').exists())
