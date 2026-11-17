"""Checkpoint/optimizer/RNG continuity and stopping tests for paired training."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph.build_graph import GraphBuilder
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.insertion import feature_names
from graph.temporal.node_token_mamba import NodeTokenMamba
from graph.temporal.pose_observations import INSERTION_CONTRACT
from training.gnn.data import TrainingBatch
from training.gnn.temporal.config import LOSS_TERMS
from training.gnn.temporal.joint_continuation import (
    capture_random_state, load_joint_continuation, restore_random_state,
)
from training.gnn.temporal import joint_finetuning
from utils.experiment_logger import ExperimentLogger


class JointContinuationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.manual_seed(12)
        self.initial_gnn = VertexEdgeGatedMPNN(16, hidden=8, layers=1)
        model_settings = dict(architecture='mamba', backend='torch_reference', d_model=8,
                              d_state=2, d_conv=2, head_hidden=4, n_layers=1)
        temporal = NodeTokenMamba(node_width=8, **model_settings)
        manifest = {'splits': {'train': ['train'], 'validation': ['dev']},
                    'parent_by_episode': {'train': 'a', 'dev': 'b'}}
        self.identity = {'dataset_sha256': 'synthetic-cache'}
        self.config = {
            'seed': 12, 'device': 'cpu', 'mode': 'joint_peg_hole', 'with_robot': True,
            'selected_m0': {'fixture': True}, 'checkpoint': str(self.root / 'gnn.pt'),
            'source_checkpoint_sha256': 'gnn-source', 'split_manifest_sha256': 'split-source',
            'temporal_split_manifest': manifest, 'source_sha256': {},
            'model': model_settings, 'node_width': 8,
            'node_token_contract': 'pre_pool_nodes_type_then_world_xyz_frame_token_v1',
            'feature_names': list(feature_names(True)), 'observation_contract': INSERTION_CONTRACT,
            'normalization': {'source_split': 'train', 'mean': [0.] * 64, 'std': [1.] * 64},
            'training': {'epochs': 4, 'sequence_length': 2, 'episode_batch_size': 1,
                         'context_policy': 'reset_windows', 'gradient_clip_norm': 1.,
                         'cuda_memory_fraction': None,
                         'early_stopping': {'patience': 30, 'min_epochs': 1, 'min_delta': 0.}},
            'loss': {'ablation': 'pose_relative_full', 'position_unit_m': .01,
                     'rotation_unit_rad': .1, 'rotation_objective': 'geodesic_huber',
                     'weights': {term: 1. for term in LOSS_TERMS}},
            'temporal_checkpoint': str(self.root / 'node.pt'),
            'joint_finetuning': {'gnn_lr': 1e-5, 'temporal_lr': 1e-4},
            'output': {'root': str(self.root / 'runs')},
        }
        torch.save(self.initial_gnn.state_dict(), self.config['checkpoint'])
        torch.save({'kind': 'node_token_mamba', 'model': temporal.state_dict(),
                    'config': deepcopy(self.config), 'dataset_sha256': 'synthetic-cache'},
                   self.config['temporal_checkpoint'])
        result = SimpleNamespace(xyz=torch.tensor([[0., 0., .1], [.02, 0., .1], [.1, .03, .1]]),
                                 object_id=torch.tensor([0, 0, 1]), metadata=[{}, {}, {}])
        graph = GraphBuilder()([result], torch.zeros(1, 3))
        target = torch.zeros(2, 21)
        target[:, 3] = 1
        target[:, 13] = 1
        target[:, 17:21] = .01
        observations = torch.randn(2, 64)
        samples = [TrainingBatch(graph, {'object21': target[:1]}, [f'frame_{i}']) for i in range(2)]
        self.windows = [(observations, target, samples), (observations + .01, target, samples)]

    def run_training(self, config, validation_loss=None):
        def prepare(effective, device, requested):
            gnn = VertexEdgeGatedMPNN(16, hidden=8, layers=1)
            gnn.load_state_dict(self.initial_gnn.state_dict())
            return gnn.eval().requires_grad_(False), {'train': None, 'validation': None}, self.identity

        original_epoch = joint_finetuning.run_epoch

        def epoch(*args, **kwargs):
            result = original_epoch(*args, **kwargs)
            optimizer = kwargs.get('optimizer', args[5] if len(args) > 5 else None)
            if validation_loss is not None and optimizer is None:
                result['loss'] = validation_loss
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.dict('os.environ', {'WANDB_MODE': 'disabled'}))
            stack.enter_context(patch.object(joint_finetuning, 'resolve_insertion_config', side_effect=deepcopy))
            stack.enter_context(patch.object(joint_finetuning, 'prepare_insertion_sequences', side_effect=prepare))
            stack.enter_context(patch.object(joint_finetuning, 'infer_insertion_data', return_value=None))
            stack.enter_context(patch.object(joint_finetuning, 'InsertionEpisodeDataset', return_value=None))
            stack.enter_context(patch.object(joint_finetuning, 'GraphWindows', return_value=self.windows))
            stack.enter_context(patch.object(joint_finetuning, 'run_epoch', side_effect=epoch))
            stack.enter_context(patch.object(joint_finetuning, 'ExperimentLogger',
                side_effect=lambda **kwargs: ExperimentLogger(**kwargs, root=self.root / 'experiments')))
            return joint_finetuning.train(config)

    def test_resume_matches_uninterrupted_weights_and_optimizer_steps(self):
        full = self.run_training(deepcopy(self.config))
        short_config = deepcopy(self.config)
        short_config['training']['epochs'] = 2
        short = self.run_training(short_config)
        resumed_config = deepcopy(self.config)
        resumed_config['continuation_checkpoint'] = str(short / 'last.pt')
        resumed = self.run_training(resumed_config)
        expected = torch.load(full / 'last.pt', weights_only=True)
        actual = torch.load(resumed / 'last.pt', weights_only=True)
        for network in ('gnn', 'temporal'):
            for name in expected[network]:
                torch.testing.assert_close(actual[network][name], expected[network][name], atol=0, rtol=0)
        self.assertEqual([row['epoch'] for row in actual['history']], list(range(5)))
        for key, value in expected['optimizer']['state'].items():
            for name, tensor in value.items():
                torch.testing.assert_close(actual['optimizer']['state'][key][name], tensor, atol=0, rtol=0)
        self.assertEqual(actual['config']['continuation']['random_state'], 'restored')

    def test_stale_counter_and_best_survive_resume_and_trigger_stopping(self):
        config = deepcopy(self.config)
        config['training']['early_stopping'] = {'patience': 3, 'min_epochs': 1, 'min_delta': 0.}
        config['training']['epochs'] = 2
        first = self.run_training(config, validation_loss=1.)
        resumed = deepcopy(config)
        resumed['training']['epochs'] = 10
        resumed['continuation_checkpoint'] = str(first / 'last.pt')
        output = self.run_training(resumed, validation_loss=1.)
        payload = torch.load(output / 'last.pt', weights_only=True)
        self.assertEqual(payload['epoch'], 3)
        self.assertEqual(payload['early_stopping_state']['stale_epochs'], 3)
        self.assertEqual(payload['best_checkpoint']['epoch'], 0)
        self.assertTrue(json.loads((output / 'training_summary.json').read_text())['early_stopped'])
        resumed['continuation_checkpoint'] = str(output / 'last.pt')
        finished = self.run_training(resumed, validation_loss=1.)
        self.assertEqual(torch.load(finished / 'last.pt', weights_only=True)['epoch'], 3)

    def test_legacy_best_keeps_optimizer_and_records_reseed(self):
        config = deepcopy(self.config)
        config['training']['epochs'] = 1
        output = self.run_training(config)
        legacy = torch.load(output / 'best.pt', weights_only=True)
        for name in ('random_state', 'history', 'early_stopping_state', 'best_checkpoint'):
            legacy.pop(name, None)
        legacy['config']['training'].pop('early_stopping', None)
        directory = self.root / 'legacy'
        directory.mkdir()
        torch.save(legacy, directory / 'best.pt')
        effective = deepcopy(legacy['config'])
        effective['training']['epochs'] = 5
        effective['training']['early_stopping'] = self.config['training']['early_stopping']
        payload, _, _, state, metadata = load_joint_continuation(directory / 'best.pt', effective, self.identity)
        self.assertIn('legacy', metadata['random_state'])
        self.assertEqual(payload['optimizer']['param_groups'], legacy['optimizer']['param_groups'])
        self.assertEqual(state['stale_epochs'], 0)
        altered = deepcopy(effective)
        altered['loss']['position_unit_m'] = .2
        with self.assertRaisesRegex(ValueError, 'loss'):
            load_joint_continuation(directory / 'best.pt', altered, self.identity)

    def test_minimum_epoch_delays_stopping(self):
        config = deepcopy(self.config)
        config['training']['epochs'] = 10
        config['training']['early_stopping'] = {'patience': 1, 'min_epochs': 3, 'min_delta': 0.}
        output = self.run_training(config, validation_loss=1.)
        payload = torch.load(output / 'last.pt', weights_only=True)
        self.assertEqual(payload['epoch'], 3)
        self.assertEqual(payload['early_stopping_state']['stale_epochs'], 3)

    def test_resume_best_does_not_keep_future_history(self):
        config = deepcopy(self.config)
        config['training']['epochs'] = 1
        output = self.run_training(config)
        payload = torch.load(output / 'best.pt', weights_only=True)
        payload.pop('history', None)
        torch.save(payload, output / 'best.pt')
        history = json.loads((output / 'history.json').read_text())
        history.append({'epoch': 2, 'train': None, 'validation': {'loss': 999.}})
        (output / 'history.json').write_text(json.dumps(history))
        effective = deepcopy(payload['config'])
        effective['training']['epochs'] = 5
        _, _, retained, _, _ = load_joint_continuation(output / 'best.pt', effective, self.identity)
        self.assertEqual([row['epoch'] for row in retained], [0, 1])
        effective['joint_finetuning']['gnn_lr'] *= 2
        with self.assertRaisesRegex(ValueError, 'gnn_lr'):
            load_joint_continuation(output / 'best.pt', effective, self.identity)

    def test_all_random_generators_restore(self):
        generator = torch.Generator().manual_seed(3)
        state = capture_random_state(generator, torch.device('cpu'))
        expected = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
        restore_random_state(state, generator, torch.device('cpu'), seed=3)
        actual = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2])
        torch.testing.assert_close(expected[3], actual[3])


if __name__ == '__main__':
    unittest.main()
