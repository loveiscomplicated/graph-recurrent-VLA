"""Check config selection, validation and model initialization metadata."""

from copy import deepcopy
import unittest

import torch

from graph.basic_gnn import BasicGNN
from graph.gated_mpnn import VertexEdgeGatedMPNN
from main import load_config
from training.gnn.trainer import build_model, resolve_config, train


class GNNTrainerTests(unittest.TestCase):
    def test_example_configs_select_models(self):
        examples = (
            ('graph/configs/train_gnn.yaml', BasicGNN),
            ('graph/configs/train_gated_mpnn.yaml', VertexEdgeGatedMPNN),
        )
        for path, expected_class in examples:
            with self.subTest(path=path):
                config = load_config(path)
                original = deepcopy(config)
                effective = resolve_config(config)
                model = build_model(effective['model'], torch.device('cpu'))
                self.assertIsInstance(model, expected_class)
                self.assertEqual(config, original)
                self.assertIn('name', effective['model'])

    def test_legacy_config_defaults_to_basic_gnn(self):
        config = load_config('graph/configs/train_gnn.yaml')
        del config['model']['name']
        effective = resolve_config(config)
        self.assertEqual(effective['model']['name'], 'basic_gnn')
        self.assertIsInstance(build_model(effective['model'], torch.device('cpu')), BasicGNN)

    def test_gated_defaults_are_recorded(self):
        config = resolve_config({'model': {
            'name': 'vertex_edge_gated_mpnn', 'num_semantic_ids': 4,
        }})
        self.assertEqual(config['model']['hidden'], 128)
        self.assertEqual(len(config['model']['size_scales_m']), 4)
        build_model(config['model'], torch.device('cpu'))

    def test_invalid_settings_fail_before_initialization(self):
        changes = (
            {'name': 'unknown'},
            {'hidden': 7},
            {'num_semantic_ids': True},
            {'edge_embedding_dim': 0},
            {'coord_scale_m': float('nan')},
            {'size_scales_m': [0.1]},
            {'size_scales_m': [0.1, 0.1, 0.1, -1]},
            {'dropout': 1.0},
            {'use_confidence': 'true'},
            {'use_semantic_ids': 1},
            {'in_dim': 3},
        )
        for change in changes:
            with self.subTest(change=change):
                config = {'model': {'name': 'vertex_edge_gated_mpnn', 'num_semantic_ids': 4}}
                config['model'].update(change)
                with self.assertRaises(ValueError):
                    resolve_config(config)


if __name__ == '__main__':
    unittest.main()
