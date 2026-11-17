"""Check config selection, validation and model initialization metadata."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from graph.basic_gnn import BasicGNN
from graph.gated_mpnn import VertexEdgeGatedMPNN
from main import load_config
from training.gnn.trainer import build_model, configure_model_compilation, resolve_config, train


class GNNTrainerTests(unittest.TestCase):
    def test_compile_settings_default_off_and_gated_recipe_enables_them(self):
        config = resolve_config(load_config('graph/configs/train_gated_mpnn.yaml'))
        self.assertEqual(config['training']['compile'], {'enabled': True, 'dynamic': True})
        del config['training']['compile']
        effective = resolve_config(config)
        self.assertEqual(effective['training']['compile'], {'enabled': False, 'dynamic': True})

    def test_invalid_compile_settings_fail_before_data_execution(self):
        for settings in (True, {'enabled': 'true'}, {'dynamic': None}, {'mode': 'reduce-overhead'}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, 'training.compile'):
                config = load_config('graph/configs/train_gated_mpnn.yaml')
                config['training']['compile'] = settings
                resolve_config(config)

    def test_compile_is_cuda_only_and_can_be_disabled(self):
        model = torch.nn.Linear(2, 1)
        for device_name, requested in (('cpu', True), ('mps', True), ('cuda', False), ('cuda', True)):
            with self.subTest(device=device_name, requested=requested):
                settings = {'enabled': requested, 'dynamic': True}
                with patch.object(model, 'compile') as compile_model:
                    metadata = configure_model_compilation(model, settings, torch.device(device_name))
                expected = requested and device_name == 'cuda'
                self.assertEqual(metadata['enabled'], expected)
                if expected:
                    compile_model.assert_called_once_with(dynamic=True, mode='default', fullgraph=False)
                else:
                    compile_model.assert_not_called()

    def test_inplace_compile_preserves_checkpoint_keys_and_optimizer_references(self):
        """Exercise real Dynamo capture on CPU; no CUDA/Inductor execution."""
        model = torch.nn.Linear(3, 2)
        expected = deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters())
        parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
        compile_function = torch.compile

        def compile_with_eager_backend(function, **settings):
            return compile_function(function, backend='eager', **settings)

        # The device selects the CUDA branch only; the trace and tensors use CPU.
        with patch('torch.compile', side_effect=compile_with_eager_backend):
            configure_model_compilation(model, {'enabled': True, 'dynamic': True}, torch.device('cuda'))
        inputs = torch.randn(4, 3)
        torch.testing.assert_close(model(inputs), expected(inputs))
        model(inputs).square().mean().backward()
        expected(inputs).square().mean().backward()
        self.assertEqual(list(model.state_dict()), list(expected.state_dict()))
        self.assertEqual([id(parameter) for parameter in model.parameters()], parameter_ids)
        restored = torch.nn.Linear(3, 2)
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(inputs), expected(inputs))
        for parameter, reference in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad)

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
