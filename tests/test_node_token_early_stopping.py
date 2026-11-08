"""Exercise early stopping and best-checkpoint selection in the actual workflow."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_temporal_pose_gru import pose_sequences
from training.gnn.temporal.node_token_mamba import train_and_evaluate


class NodeTokenEarlyStoppingTest(unittest.TestCase):
    def test_training_stops_and_evaluates_the_best_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'source.pt'
            checkpoint.touch()
            selection = root / 'selection.jsonl'
            selection.write_text('{}\n')
            effective = {
                'device': 'cpu', 'seed': 0, 'checkpoint': str(checkpoint),
                'evaluation_selection': str(selection), 'output': {'root': str(root / 'output')},
                'training': {'epochs': 20, 'lr': .001, 'sequence_length': 4,
                             'cuda_memory_fraction': None,
                             'early_stopping': {'patience': 2, 'min_epochs': 3, 'min_delta': 0.0}},
                'mode': 'joint_peg_hole', 'with_robot': True,
                'model': {'architecture': 'mamba', 'backend': 'torch_reference',
                          'd_model': 16, 'n_layers': 1, 'd_state': 4, 'd_conv': 2,
                          'expand': 2, 'head_hidden': 8, 'position_mode': 'residual',
                          'position_scale_m': .01},
                'source_sha256': {},
            }
            data = pose_sequences(with_robot=True)
            data['node_features'] = [np.ones((2, 5), dtype=np.float32) for _ in data['frame']]
            identity = {'dataset_sha256': 'fixture'}

            def prepare(config, device, requested):
                return torch.nn.Identity(), {name: object() for name in requested}, identity

            losses = [1., 3., 1., 2., 1., 2., 1., 3.]
            with patch.dict(os.environ, {'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(root / 'runs')}), \
                 patch('training.gnn.temporal.node_token_mamba.resolve_insertion_config', return_value=effective), \
                 patch('training.gnn.temporal.node_token_mamba.prepare_insertion_sequences', side_effect=prepare), \
                 patch('training.gnn.temporal.node_token_mamba.infer_node_data', return_value=data), \
                 patch('training.gnn.temporal.node_token_mamba.run_epoch', side_effect=losses), \
                 patch('training.gnn.temporal.node_token_mamba.evaluate_selected') as evaluate:
                output = train_and_evaluate({})
            history = json.loads((output / 'history.json').read_text())
            best = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
            last = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual([row['epoch'] for row in history], [1, 2, 3, 4])
            self.assertEqual(best['epoch'], 2)
            self.assertEqual(last['early_stopping_state']['stale_epochs'], 2)
            evaluate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
