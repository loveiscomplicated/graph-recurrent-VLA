"""Baseline equivalence, component replacement and stream/training lifecycle."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from controller import RNNPolicy
from gnn import GraphEncoder
from vla.contracts import Graph, pack_graphs
from vla.integration import CachedGraphPolicy, EpisodeStream
from vla.integration.live_policy import gradient_report
from vla.workflows.components import LiveComponents, components_from_config
from vla.data.files import tensor_identity
from vla.state import map_state
from vla.recording import snapshot_state, state_history_arrays
from graph.integration_baseline.robot import RobotNormalizer
from graph.prototype.v2.state import StateNormalizer
from graph.integration_baseline.bc08_codec import ActionCodec


class SmallEncoder(nn.Module):
    output_dim = 9

    def __init__(self):
        super().__init__()
        self.project = nn.Linear(21, self.output_dim)

    def forward(self, graphs, trace=None):
        encoded = self.project(graphs.node_input)
        pooled = encoded.new_zeros((graphs.count, self.output_dim)).index_add(0, graphs.node_graph, encoded)
        return pooled, encoded.new_zeros((graphs.count, 2))


class StatelessController(nn.Module):
    action_dim = 7
    state_batch_dim = 0

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.head = nn.Linear(input_dim, self.action_dim)

    def forward(self, sequence, state=None):
        return self.head(sequence), None


def encoder_factory(config):
    return SmallEncoder()


def controller_factory(input_dim, config):
    return StatelessController(input_dim)


class ModularLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((Path(__file__).parent/'fixtures/live_policy_d222fe2.json').read_text())

    def setUp(self):
        torch.set_num_threads(1)
        self.norm = RobotNormalizer({'dimension': 76}, StateNormalizer(torch.zeros(76), torch.ones(76)), [1])
        self.graphs = [Graph(**{key: torch.tensor(value, dtype=torch.long if key in
            ('node_type', 'edge_index', 'edge_type') else torch.float32) for key, value in row.items()},
            packet={'coordinate_frame': 'EE_metres'}, timings_ms={}) for row in self.reference['inputs']['graphs']]
        for graph in self.graphs:
            graph.edge_attr = graph.edge_attr.reshape(-1, 4)
        self.robot = torch.tensor(self.reference['inputs']['robot'])
        self.config = dict(d=8, L=2, gru_hidden=12, gru_layers=1, enc_hidden=10)

    def test_pre_refactor_weights_outputs_state_and_gradients(self):
        for variant, expected in self.reference['reference'].items():
            torch.manual_seed(self.reference['seed'])
            model = LiveComponents().model(self.norm.schema, variant, self.reference['config'])
            self.assertEqual(list(model.state_dict()), expected['keys'])
            self.assertEqual(tensor_identity(model.state_dict()), expected['state_sha256'])
            pred, state = model(pack_graphs(self.graphs, self.reference['inputs']['frame_map']), self.robot, self.norm)
            torch.testing.assert_close(pred, torch.tensor(expected['action']), atol=0, rtol=0)
            torch.testing.assert_close(state, torch.tensor(expected['hidden']), atol=0, rtol=0)
            pred.square().mean().backward()
            self.assertEqual(gradient_report(model), expected['gradients'])

    def test_legacy_imports_share_the_canonical_implementation(self):
        from graph.integration_baseline.model import GraphEncoder as OldEncoder
        from recurrent.models import RNNPolicy as OldController
        from graph.plan_a_bc.model import CachedGraphPolicy as OldPolicy
        from graph.integration_baseline.graph import Graph as OldGraph
        from graph.plan_a_bc import rollout, cache, train
        from vla.workflows import rollout as new_rollout, training
        from vla.data import graph_cache
        self.assertIs(OldEncoder, GraphEncoder)
        self.assertIs(OldController, RNNPolicy)
        self.assertIs(OldPolicy, CachedGraphPolicy)
        self.assertIs(OldGraph, Graph)
        self.assertIs(rollout, new_rollout)
        self.assertIs(train, training)
        self.assertIs(cache, graph_cache)

    def test_custom_encoder_dimension_and_stateless_controller_train(self):
        components = components_from_config({'components': {
            'graph_encoder': 'tests.test_modular_live:encoder_factory',
            'controller': 'tests.test_modular_live:controller_factory'}})
        model = components.model(self.norm.schema, 'E0', self.config)
        self.assertEqual(model.fused_dim, 9+2+76)
        self.assertEqual(model.state_batch_dim, 0)
        pred, state = model(pack_graphs(self.graphs, [[0,0],[0,1],[1,0]]), self.robot, self.norm)
        self.assertIsNone(state)
        self.assertTrue(torch.equal(pred[1,1], torch.zeros(7)))
        pred.sum().backward()
        gradients = gradient_report(model)
        self.assertGreater(gradients['encoder'], 0)
        self.assertGreater(gradients['controller'], 0)
        self.assertNotIn('gru', gradients)

    def test_lstm_state_reset_and_padded_streams(self):
        components = LiveComponents(controller=lambda width, cfg:
            RNNPolicy(width, 7, cell='lstm', hidden=9, enc_hidden=5))
        model = components.model(self.norm.schema, 'E0', self.config).eval()
        pred, state = model(pack_graphs(self.graphs, [[0,0],[0,1],[1,0]]), self.robot, self.norm)
        self.assertIsInstance(state, tuple)
        self.assertEqual(state[0].shape, (1,2,9))
        stream = EpisodeStream(model, self.norm)
        one = pack_graphs(self.graphs[:1], [(0,0)])
        first = stream.step(one, self.robot[:1], 'one', 0)
        stream.step(one, self.robot[:1], 'one', 1)
        reset = stream.step(one, self.robot[:1], 'two', 0)
        torch.testing.assert_close(first, reset, atol=0, rtol=0)

    def test_training_chunks_handle_stateless_and_lstm_controllers(self):
        from vla.workflows.training import train_update
        graphs, robot = self.graphs[:1], self.robot[:1]
        class Dataset:
            def batch(self, ids, start, length, device):
                return pack_graphs(graphs, [(0,0)]), robot.to(device), torch.zeros(1,1,7), torch.ones(1,1,dtype=torch.bool)
        codec = ActionCodec.fit('B', {1: np.random.default_rng(42).normal(size=(10,7)).astype('float32')}, [1])
        for factory in (controller_factory, lambda width, cfg: RNNPolicy(width,7,cell='lstm',hidden=9,enc_hidden=5)):
            model = LiveComponents(controller=factory).model(self.norm.schema, 'E0', self.config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            progress = {'hidden': {}}
            config = {'training': {'sequence_length': 1, 'gradient_clip': 1.}}
            for start in (0,1):
                result = train_update(model,optimizer,Dataset(),self.norm,codec,progress,[1],start,config,'cpu')
                self.assertTrue(np.isfinite(result['optimization_loss']))
                map_state(lambda value: self.assertFalse(value.requires_grad), progress['hidden']['1'])

    def test_graph_builder_is_selected_with_coordinate_frame(self):
        calls = []
        def factory(assets, schema, **kwargs):
            calls.append((assets,schema,kwargs))
            return SimpleNamespace(build=lambda *a: None, from_h5=lambda *a: None,
                                   reset=lambda: None, close=lambda: None)
        result = LiveComponents(graph_builder=factory).geometry('assets', self.norm.schema, coordinate_frame='world')
        self.assertTrue(callable(result.build))
        self.assertEqual(calls[0][2], {'coordinate_frame': 'world'})

    def test_offline_evaluation_reads_real_cache_and_enforces_test_gate(self):
        from vla.workflows.evaluation import evaluate
        from vla.data.graph_cache import input_identity, GRAPH_KEYS
        from vla.data.files import write, sha, read
        import os
        codec = ActionCodec.fit('B', {1: np.random.default_rng(42).normal(size=(10,7)).astype('float32')}, [1])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            root = Path(tmp)
            config = dict(seed=42, geometry={}, source_sha256={}, model=self.config,
                training={'sequence_length': 1}, splits={'train': [1], 'validation': [2], 'test': [3]})
            write(root/'config.json', config)
            write(root/'normalizer.json', dict(schema=self.norm.schema, mean=[0.]*76,
                std=[1.]*76, fitted_episode_ids=[1]))
            write(root/'action_codec.json', codec.state)
            write(root/'dataset_audit.json', dict(dataset_sha256='fixture', split_sha256='fixture'))
            graph = self.graphs[0]
            arrays = {key: getattr(graph,key).numpy() for key in GRAPH_KEYS}
            arrays.update(node_ptr=np.array([0,len(graph.node_type)]), edge_ptr=np.array([0,graph.edge_index.shape[1]]),
                robot76=self.robot[:1].numpy(), action7=np.zeros((1,7),np.float32),
                presence2=np.zeros((1,2),np.float32), timestep=np.array([0]), coordinate_frame=np.array('EE_metres'))
            np.savez(root/'graphs.npz', **arrays)
            write(root/'cache_manifest.json', dict(identity=input_identity(root), episodes=[
                dict(episode_id=i, split=split, actions=1, path='graphs.npz', sha256=sha(root/'graphs.npz'))
                for i,split in ((1,'train'),(2,'validation'),(3,'test'))]))
            checkpoint = root/'checkpoint.pt'
            model = LiveComponents().model(self.norm.schema, 'E0', self.config)
            torch.save(dict(config_sha256=sha(root/'config.json'), normalizer=read(root/'normalizer.json'),
                codec=codec.state, components=LiveComponents().identity(), variant='E0',
                model=model.state_dict(), cache_manifest_sha256=sha(root/'cache_manifest.json')), checkpoint)
            path = evaluate(root,checkpoint,split='validation',device='cpu')
            self.assertEqual(read(path)['actions'], 1)
            self.assertTrue((path.parent/'predictions/ep2.npz').is_file())
            self.assertEqual(read(path.parent/'metadata.json')['status'], 'finished')
            with self.assertRaisesRegex(ValueError, 'test gate locked'):
                evaluate(root,checkpoint,split='test',device='cpu')
            write(root/'model_selection_frozen.json', {'status':'fixture'})
            path = evaluate(root,checkpoint,split='test',device='cpu')
            self.assertTrue((path.parent/'predictions/ep3.npz').is_file())

    def test_component_checkpoint_identity_cannot_silently_change(self):
        baseline = LiveComponents()
        baseline.validate_checkpoint({})
        changed = LiveComponents(graph_encoder=encoder_factory)
        with self.assertRaisesRegex(ValueError, 'lacks identity'):
            changed.validate_checkpoint({})
        with self.assertRaisesRegex(ValueError, 'selection changed'):
            changed.validate_checkpoint({'components': baseline.identity()})
        changed.validate_checkpoint({'components': changed.identity()})

    def test_invalid_factories_and_dimensions_fail_explicitly(self):
        for values in ({'unknown': 'x:y'}, {'graph_encoder': 'not_a_path'},
                       {'graph_encoder': 'vla.nn:missing'}):
            with self.assertRaises((ValueError,AttributeError)):
                components_from_config({'components':values})
        with self.assertRaises(TypeError):
            LiveComponents(graph_encoder=lambda cfg:None).model(self.norm.schema,'E0',self.config)
        with self.assertRaisesRegex(ValueError, 'input dimension'):
            CachedGraphPolicy(self.norm.schema,'E0',**self.config,controller=StatelessController(1))

    def test_state_archival_keeps_gru_format_and_supports_other_states(self):
        state = torch.randn(1,1,12)
        history = [snapshot_state(state), snapshot_state(state+1)]
        result = state_history_arrays(history)
        self.assertEqual(set(result), {'hidden'})
        np.testing.assert_array_equal(result['hidden'], np.asarray(history))
        result = state_history_arrays([snapshot_state((state,state+1))]*2)
        self.assertIn('hidden_0', result)
        self.assertIn('hidden_1', result)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            np.savez(path, **result)
            with np.load(path,allow_pickle=False) as saved:
                self.assertEqual(saved['hidden_0'].shape,(2,1,1,12))
        self.assertEqual(state_history_arrays([None,None])['hidden'].shape,(2,0))


class GroundingPolicyTests(unittest.TestCase):
    def test_default_has_no_auth_fallback(self):
        from vla.perception.grounding import grounding_from_config, grounding_identity
        from visual.runtime.live_grounding import AuthenticationError
        with patch.dict('os.environ', {'OPENAI_API_KEY':''}, clear=True), \
                patch('visual.runtime.live_grounding.load_env_file',return_value=False), \
                patch('visual.runtime.live_grounding.GDino') as dino:
            with self.assertRaises(AuthenticationError):grounding_from_config(device='cpu')
            dino.assert_not_called()
        self.assertIsNone(grounding_identity()['fallback'])

    def test_auth_error_is_not_converted_to_dino_in_default_mode(self):
        from vla.perception.grounding import grounding_from_config
        from visual.runtime.live_grounding import AuthenticationError
        with patch.dict('os.environ', {'OPENAI_API_KEY':'test-key'}, clear=True), \
                patch('visual.runtime.live_grounding.load_env_file',return_value=False), \
                patch('visual.runtime.live_grounding.GDino') as dino:
            grounder = grounding_from_config(device='cpu')
            with patch.object(grounder,'_request',side_effect=AuthenticationError('http_401')):
                with self.assertRaises(AuthenticationError):grounder.initialize([np.zeros((16,16,3),np.uint8)])
            self.assertEqual(grounder.backend,'terra')
            dino.assert_not_called()

    def test_preserved_fallback_is_an_explicit_choice(self):
        from vla.perception.grounding import grounding_from_config, grounding_identity
        config = {'grounding': {'policy':'terra_auth_fallback'}}
        with patch.dict('os.environ', {'OPENAI_API_KEY':''}, clear=True), \
                patch('visual.runtime.live_grounding.load_env_file',return_value=False):
            grounder = grounding_from_config(config,device='cpu')
        self.assertEqual(grounder.backend,'dino')
        self.assertEqual(grounding_identity(config)['fallback'],'grounding_dino')


if __name__ == '__main__':
    unittest.main()
